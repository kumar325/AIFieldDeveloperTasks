"""End-to-end tests that run server.py as a subprocess and send raw JSON-RPC."""

from __future__ import annotations

import ast
import json
import os
import queue
import subprocess
import threading
import sys
from pathlib import Path

import pytest

SERVER = Path(__file__).resolve().parent.parent / "server.py"
TIMEOUT = 10


class RawClient:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self.stdout_lines: list[str] = []
        self._next_id = 100
        # select() doesn't work on Windows pipes.
        self._lines: queue.Queue[bytes] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            self._lines.put(raw)
        self._lines.put(b"")  # EOF marker

    def send_raw(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((line + "\n").encode())
        self.proc.stdin.flush()

    def read_frame(self) -> dict:
        try:
            raw = self._lines.get(timeout=TIMEOUT)
        except queue.Empty:
            raise AssertionError("timed out waiting for a response on stdout") from None
        assert raw, "server closed stdout"
        line = raw.decode("utf-8")
        assert not line.endswith("\r\n"), "stdout frame has CRLF line ending"
        self.stdout_lines.append(line)
        frame = json.loads(line)
        assert frame.get("jsonrpc") == "2.0", f"not a JSON-RPC 2.0 frame: {line!r}"
        return frame

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        msg = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send_raw(json.dumps(msg))
        frame = self.read_frame()
        assert frame["id"] == self._next_id
        return frame

    def call(self, tool: str, arguments: dict | None) -> dict:
        params: dict = {"name": tool}
        if arguments is not None:
            params["arguments"] = arguments
        return self.request("tools/call", params)

    def close(self) -> tuple[str, str]:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read() if self.proc.stderr else b""
        self.proc.wait(timeout=TIMEOUT)
        self._reader.join(timeout=TIMEOUT)
        rest = []
        while not self._lines.empty():
            raw = self._lines.get()
            if raw:
                rest.append(raw.decode("utf-8"))
        return "".join(rest), err.decode("utf-8", errors="replace")


@pytest.fixture
def client():
    c = RawClient()
    init = c.request(
        "initialize",
        {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
    )
    assert "result" in init
    c.send_raw(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
    yield c
    if c.proc.poll() is None:
        c.close()


def assert_invalid_params(frame: dict, field: str | None = None) -> None:
    assert "error" in frame, f"expected a JSON-RPC error, got {frame}"
    assert frame["error"]["code"] == -32602
    if field is not None:
        fields = [e["field"] for e in frame["error"]["data"]["errors"]]
        assert field in fields, f"expected error on {field}, got {fields}"


VALID_REFUND = {"customer_id": "CUST-00001", "amount": 25.5, "reason": "Item arrived damaged"}


def test_stdout_contains_only_jsonrpc_and_logs_go_to_stderr(client):
    client.request("tools/list")
    client.call("get_customer_record", {"customer_id": "CUST-12345"})
    client.call("get_customer_record", {"customer_id": "bad"})
    client.send_raw("garbage")
    client.read_frame()
    out, err = client.close()
    all_lines = client.stdout_lines + [line for line in out.splitlines(keepends=True) if line]
    for line in all_lines:
        assert line.endswith("\n")
        assert json.loads(line)["jsonrpc"] == "2.0"
    assert "starting on stdio" in err
    assert "get_customer_record ok" in err


def test_stray_writes_after_guard_never_reach_stdout():
    code = (
        "import sys; sys.path.insert(0, %r); import server, os, subprocess; "
        "print('LEAK-print'); sys.stdout.write('LEAK-sys\\n'); os.write(1, b'LEAK-fd1\\n'); "
        "subprocess.run([sys.executable, '-c', 'print(\\'LEAK-child\\')'])"
    ) % str(SERVER.parent)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=TIMEOUT)
    assert proc.stdout == b""
    for marker in (b"LEAK-print", b"LEAK-sys", b"LEAK-fd1", b"LEAK-child"):
        assert marker in proc.stderr


def test_source_has_no_stdout_writes():
    tree = ast.parse(SERVER.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            offenders.append(f"print() at line {node.lineno}")
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "write"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "stdout"
        ):
            offenders.append(f"sys.stdout.write at line {node.lineno}")
    assert not offenders, offenders


def test_tools_list_advertises_strict_schemas(client):
    tools = {t["name"]: t for t in client.request("tools/list")["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}
    refund = tools["trigger_refund"]["inputSchema"]
    assert refund["additionalProperties"] is False
    assert set(refund["required"]) == {"customer_id", "amount", "reason"}
    assert refund["properties"]["amount"]["exclusiveMinimum"] == 0
    assert refund["properties"]["reason"]["minLength"] == 10
    assert refund["properties"]["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"


def test_parse_error_gets_32700_with_null_id(client):
    client.send_raw("{this is not json")
    frame = client.read_frame()
    assert frame["id"] is None
    assert frame["error"]["code"] == -32700


@pytest.mark.parametrize("line", ['{"foo": 1}', "[1, 2]", '{"jsonrpc": "1.0", "id": 1, "method": "ping"}'])
def test_invalid_request_gets_32600_with_null_id(client, line):
    client.send_raw(line)
    frame = client.read_frame()
    assert frame["id"] is None
    assert frame["error"]["code"] == -32600


def test_unknown_method_gets_32601(client):
    assert client.request("does/not/exist")["error"]["code"] == -32601


def test_unknown_tool_gets_32602(client):
    frame = client.call("delete_everything", {})
    assert frame["error"]["code"] == -32602
    assert "Unknown tool" in frame["error"]["message"]


def test_server_survives_a_burst_of_bad_input(client):
    client.send_raw("nope")
    client.read_frame()
    client.call("trigger_refund", {"amount": "lots"})
    client.call("nope", None)
    ok = client.call("get_customer_record", {"customer_id": "CUST-00001"})
    assert ok["result"]["isError"] is False


def test_get_customer_record_ok(client):
    result = client.call("get_customer_record", {"customer_id": "CUST-12345"})["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["name"] == "Alan Turing"
    assert json.loads(result["content"][0]["text"])["customer_id"] == "CUST-12345"


def test_trigger_refund_ok_and_balance_decreases(client):
    r = client.call("trigger_refund", VALID_REFUND)["result"]
    assert r["isError"] is False
    assert r["structuredContent"]["status"] == "issued"
    assert r["structuredContent"]["remaining_refundable_balance"] == 474.5


def test_integer_amount_is_accepted(client):
    r = client.call("trigger_refund", {**VALID_REFUND, "amount": 10})["result"]
    assert r["isError"] is False


def test_unknown_customer_is_tool_error_not_protocol_error(client):
    frame = client.call("get_customer_record", {"customer_id": "CUST-55555"})
    assert "error" not in frame
    assert frame["result"]["isError"] is True
    assert "not found" in frame["result"]["content"][0]["text"]


def test_refund_over_balance_is_tool_error(client):
    frame = client.call("trigger_refund", {**VALID_REFUND, "customer_id": "CUST-99999"})
    assert frame["result"]["isError"] is True


@pytest.mark.parametrize(
    "bad_id",
    [
        "CUST-1234",  # too short
        "CUST-123456",  # too long
        "cust-12345",  # case
        "CUST_12345",  # separator
        "CUST-ABCDE",  # letters
        " CUST-12345",  # leading space
        "CUST-12345 ",  # trailing space
        "CUST-12345\n",  # trailing newline ($ edge case)
        "xxCUST-12345",  # unanchored prefix
        "CUST-١٢٣٤٥",  # Arabic-Indic digits (\d edge case)
        "",
    ],
)
def test_bad_customer_id_format(client, bad_id):
    assert_invalid_params(client.call("get_customer_record", {"customer_id": bad_id}), "customer_id")


@pytest.mark.parametrize("bad_value", [12345, None, True, ["CUST-12345"], {"id": "CUST-12345"}])
def test_customer_id_wrong_type(client, bad_value):
    assert_invalid_params(client.call("get_customer_record", {"customer_id": bad_value}), "customer_id")


def test_missing_customer_id(client):
    assert_invalid_params(client.call("get_customer_record", {}), "customer_id")


def test_missing_arguments_object(client):
    assert_invalid_params(client.call("get_customer_record", None), "customer_id")


def test_extra_field_rejected(client):
    frame = client.call("get_customer_record", {"customer_id": "CUST-12345", "admin": True})
    assert_invalid_params(frame, "admin")


@pytest.mark.parametrize(
    "bad_amount",
    [0, 0.0, -1, -0.01, 10_000.01, 1e300, 0.001, 12.345, "10.00", "10", True, False, None, [10]],
)
def test_bad_amount(client, bad_amount):
    assert_invalid_params(client.call("trigger_refund", {**VALID_REFUND, "amount": bad_amount}), "amount")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_amount(client, literal):
    # Not valid JSON, but the SDK parser accepts these literals.
    line = (
        '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"trigger_refund",'
        f'"arguments":{{"customer_id":"CUST-00001","amount":{literal},"reason":"Item arrived damaged"}}}}}}'
    )
    client.send_raw(line)
    frame = client.read_frame()
    assert_invalid_params(frame, "amount")


def test_max_amount_boundary_ok(client):
    frame = client.call("trigger_refund", {**VALID_REFUND, "amount": 10_000})
    assert "error" not in frame


@pytest.mark.parametrize(
    "bad_reason",
    [
        "too short",  # 9 chars
        " " * 20,  # whitespace only
        "   short    ",  # padded to >=10 but <10 real chars
        "x" * 501,  # too long
        "valid text\x00here",  # control character
        1234567890,
        None,
    ],
)
def test_bad_reason(client, bad_reason):
    assert_invalid_params(client.call("trigger_refund", {**VALID_REFUND, "reason": bad_reason}), "reason")


def test_reason_exactly_ten_chars_ok(client):
    frame = client.call("trigger_refund", {**VALID_REFUND, "reason": "0123456789"})
    assert frame["result"]["isError"] is False


def test_unicode_reason_ok(client):
    frame = client.call("trigger_refund", {**VALID_REFUND, "reason": "Producto dañado en envío 📦"})
    assert frame["result"]["isError"] is False


def test_multiple_errors_reported_together(client):
    frame = client.call("trigger_refund", {"customer_id": "bad", "amount": -1, "reason": "no"})
    assert_invalid_params(frame)
    fields = {e["field"] for e in frame["error"]["data"]["errors"]}
    assert fields == {"customer_id", "amount", "reason"}
