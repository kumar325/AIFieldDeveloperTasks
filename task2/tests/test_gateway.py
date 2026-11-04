"""Gateway tests: in-process against the mock, plus end-to-end over real HTTP."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import jwt
import pytest
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gateway  # noqa: E402
import mock_server  # noqa: E402

SECRET = "test-secret-that-is-at-least-32-characters"


def token(role="viewer", sub="tester", ttl=300, secret=SECRET, **extra):
    now = int(time.time())
    claims = {"sub": sub, "role": role, "iat": now, "exp": now + ttl, **extra}
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, secret, algorithm="HS256")


def auth(role="viewer", **kw):
    return {"Authorization": f"Bearer {token(role, **kw)}"}


@pytest.fixture
def mock():
    return mock_server.create_app()


@pytest.fixture
def client(mock):
    app = gateway.create_app("http://downstream/mcp", SECRET, transport=httpx.ASGITransport(app=mock))
    with TestClient(app) as c:
        yield c


def rpc(method, params=None, msg_id=1):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def call(name, arguments=None, msg_id=1):
    return rpc("tools/call", {"name": name, "arguments": arguments or {}}, msg_id)


def post(client, body, headers=None, role="viewer"):
    h = auth(role) if headers is None else headers
    if isinstance(body, (dict, list)):
        return client.post("/mcp", json=body, headers=h)
    return client.post("/mcp", content=body, headers={**h, "Content-Type": "application/json"})


def downstream_methods(mock):
    return [c["message"].get("method") for c in mock.state.calls]


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": ""},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer   "},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer not.a.jwt"},
        {"Authorization": f"Bearer {token(secret='a-different-secret-also-32-chars-long')}"},
        {"Authorization": f"Bearer {token(ttl=-10)}"},
        {"Authorization": f"Bearer {token(role=None)}"},
        {"Authorization": f"Bearer {token(role='superuser')}"},
        {"Authorization": f"Bearer {token(role='Admin')}"},
        {"Authorization": f"Bearer {token(role=['admin'])}"},
        {"Authorization": f"Bearer {token(sub=None)}"},
    ],
)
def test_bad_credentials_get_401_and_never_reach_downstream(client, mock, headers):
    r = post(client, rpc("tools/list"), headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"].startswith("Bearer")
    assert r.json()["error"]["code"] == -32000
    assert mock.state.calls == []


def test_alg_none_token_rejected(client, mock):
    unsigned = jwt.encode({"sub": "x", "role": "admin", "exp": int(time.time()) + 60}, None, algorithm="none")
    r = post(client, call("admin_reset_key"), headers={"Authorization": f"Bearer {unsigned}"})
    assert r.status_code == 401
    assert mock.state.calls == []


def test_bearer_scheme_is_case_insensitive(client):
    r = post(client, rpc("tools/list"), headers={"Authorization": f"bearer {token()}"})
    assert r.status_code == 200


def test_get_and_delete_require_auth(client):
    assert client.get("/mcp").status_code == 401
    assert client.delete("/mcp").status_code == 401
    assert client.get("/mcp", headers=auth()).status_code == 405  # mock has no GET stream
    assert client.delete("/mcp", headers=auth()).status_code == 200


def test_tools_list_forwarded_transparently(client, mock):
    r = post(client, rpc("tools/list", msg_id="abc"))
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "abc"
    assert {t["name"] for t in body["result"]["tools"]} == {t["name"] for t in mock_server.TOOLS}
    assert mock.state.calls[0]["message"] == rpc("tools/list", msg_id="abc")


def test_client_token_is_not_passed_downstream(client, mock):
    post(client, rpc("tools/list"), headers={**auth(), "X-Custom": "1", "Cookie": "a=b"})
    sent = mock.state.calls[0]["headers"]
    assert "authorization" not in sent
    assert "cookie" not in sent
    assert "x-custom" not in sent


def test_session_headers_forwarded_both_ways(client, mock):
    init = post(client, rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                           "clientInfo": {"name": "t", "version": "0"}}))
    session = init.headers["mcp-session-id"]
    assert session
    post(client, rpc("ping", msg_id=2),
         headers={**auth(), "Mcp-Session-Id": session, "MCP-Protocol-Version": "2025-11-25"})
    sent = mock.state.calls[-1]["headers"]
    assert sent["mcp-session-id"] == session
    assert sent["mcp-protocol-version"] == "2025-11-25"


def test_notification_forwarded_with_202(client, mock):
    r = post(client, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert r.status_code == 202
    assert downstream_methods(mock) == ["notifications/initialized"]


def test_client_response_message_forwarded(client, mock):
    r = post(client, {"jsonrpc": "2.0", "id": 7, "result": {"roots": []}})
    assert r.status_code == 202
    assert mock.state.calls[0]["message"]["id"] == 7


def test_downstream_errors_pass_through_unchanged(client):
    r = post(client, rpc("does/not/exist"))
    assert r.json()["error"]["code"] == -32601


def test_sse_response_streamed_through(client):
    r = post(client, call("stream_status"), headers={**auth(), "Accept": "application/json, text/event-stream"})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[0]["method"] == "notifications/progress"
    assert events[1]["result"]["content"][0]["text"] == "done"


def test_downstream_unavailable_returns_502_with_request_id():
    app = gateway.create_app("http://127.0.0.1:1/mcp", SECRET)
    with TestClient(app) as c:
        r = post(c, rpc("tools/list", msg_id=42))
    assert r.status_code == 502
    assert r.json() == {"jsonrpc": "2.0", "id": 42,
                        "error": {"code": -32603, "message": "Downstream server unavailable"}}


def test_viewer_can_call_regular_tool(client, mock):
    r = post(client, call("get_order", {"order_id": "A1"}))
    assert r.json()["result"]["isError"] is False
    assert downstream_methods(mock) == ["tools/call"]


def test_viewer_blocked_from_admin_tool(client, mock):
    r = post(client, call("admin_reset_key", {"customer_id": "C1"}, msg_id=9))
    assert r.status_code == 200
    assert r.json() == {
        "jsonrpc": "2.0",
        "id": 9,
        "error": {"code": -32001, "message": "Unauthorized Tool Call",
                  "data": {"tool": "admin_reset_key", "required_role": "admin"}},
    }
    assert mock.state.calls == []


def test_admin_can_call_admin_tool(client, mock):
    r = post(client, call("admin_reset_key", {"customer_id": "C1"}), role="admin")
    assert r.json()["result"]["content"][0]["text"] == "key rotated for C1"
    assert downstream_methods(mock) == ["tools/call"]


@pytest.mark.parametrize("name", ["ADMIN_reset_key", "Admin_delete_user", "admin_", "admin_x.y"])
def test_admin_prefix_check_ignores_case(client, mock, name):
    assert post(client, call(name)).json()["error"]["code"] == -32001
    assert mock.state.calls == []


@pytest.mark.parametrize("name", ["admin", "administrator_tool", "get_admin_report", "adminreset"])
def test_names_without_admin_underscore_prefix_are_not_admin_tools(client, mock, name):
    # Reaches downstream, which reports the unknown tool.
    r = post(client, call(name))
    assert r.json()["error"]["code"] == -32602
    assert downstream_methods(mock) == ["tools/call"]


@pytest.mark.parametrize(
    "name",
    [" admin_reset_key", "admin_reset_key ", "​admin_reset_key", "ａｄｍｉｎ_reset_key",
     "admin＿reset_key", "admin_reset_key\n", "", "x" * 129],
)
def test_suspicious_tool_names_rejected_before_downstream(client, mock, name):
    r = post(client, call(name))
    assert r.json()["error"]["code"] == -32602
    assert mock.state.calls == []


@pytest.mark.parametrize(
    "params",
    [None, {}, {"name": None}, {"name": 123}, {"name": ["admin_reset_key"]}, ["admin_reset_key"]],
)
def test_tools_call_without_string_name_rejected(client, mock, params):
    msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}
    if params is not None:
        msg["params"] = params
    r = post(client, msg)
    assert r.json()["error"]["code"] == -32602
    assert mock.state.calls == []


def test_tools_call_notification_rejected(client, mock):
    r = post(client, {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "admin_reset_key"}})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32600
    assert mock.state.calls == []


def test_duplicate_name_key_smuggling_blocked(client, mock):
    body = ('{"jsonrpc":"2.0","id":1,"method":"tools/call",'
            '"params":{"name":"admin_reset_key","name":"get_order"}}')
    r = post(client, body)
    assert r.json()["error"]["code"] == -32600
    assert mock.state.calls == []


def test_duplicate_method_key_smuggling_blocked(client, mock):
    body = '{"jsonrpc":"2.0","id":1,"method":"tools/list","method":"tools/call","params":{"name":"admin_reset_key"}}'
    assert post(client, body).json()["error"]["code"] == -32600
    assert mock.state.calls == []


def test_batch_cannot_hide_admin_call(client, mock):
    r = post(client, [rpc("tools/list"), call("admin_reset_key", msg_id=2)])
    assert r.status_code == 400
    assert r.json()["error"] == {"code": -32600, "message": "Batch requests are not supported"}
    assert mock.state.calls == []


@pytest.mark.parametrize("body", ["{not json", "", '{"jsonrpc":"2.0","id":1,"method":"ping",}', "\xff\xfe"])
def test_parse_error(client, mock, body):
    r = post(client, body.encode("latin-1"))
    assert r.status_code == 400
    assert r.json() == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
    assert mock.state.calls == []


def test_nan_literal_is_parse_error(client):
    r = post(client, b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_order","x":NaN}}')
    assert r.json()["error"]["code"] == -32700


@pytest.mark.parametrize(
    "body",
    [
        '"just a string"',
        "42",
        "null",
        '{"id":1,"method":"tools/list"}',
        '{"jsonrpc":"1.0","id":1,"method":"tools/list"}',
        '{"jsonrpc":2.0,"id":1,"method":"tools/list"}',
        '{"jsonrpc":"2.0","id":1,"method":5}',
        '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":"x"}',
        '{"jsonrpc":"2.0","id":true,"method":"tools/list"}',
        '{"jsonrpc":"2.0","id":{"a":1},"method":"tools/list"}',
        '{"jsonrpc":"2.0","id":1,"method":"tools/list","result":{}}',
        '{"jsonrpc":"2.0","id":1}',
        '{"jsonrpc":"2.0","id":1,"result":{},"error":{"code":1,"message":"x"}}',
        '{"jsonrpc":"2.0","result":{}}',
    ],
)
def test_invalid_request(client, mock, body):
    r = post(client, body)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == -32600
    assert mock.state.calls == []


def test_error_echoes_request_id_when_valid(client):
    assert post(client, call("admin_x", msg_id="req-7")).json()["id"] == "req-7"


def test_wrong_content_type_rejected(client, mock):
    r = client.post("/mcp", content=json.dumps(rpc("tools/list")),
                    headers={**auth(), "Content-Type": "text/plain"})
    assert r.status_code == 415
    assert mock.state.calls == []


def test_content_type_with_charset_accepted(client):
    r = client.post("/mcp", content=json.dumps(rpc("tools/list")),
                    headers={**auth(), "Content-Type": "application/json; charset=utf-8"})
    assert r.status_code == 200


def test_oversized_body_rejected(client, mock):
    big = call("get_order", {"pad": "x" * (gateway.MAX_BODY_BYTES + 1)})
    r = post(client, big)
    assert r.status_code == 413
    assert mock.state.calls == []


def test_unicode_arguments_forwarded_intact(client, mock):
    post(client, call("get_order", {"order_id": "注文-📦"}))
    assert mock.state.calls[0]["message"]["params"]["arguments"]["order_id"] == "注文-📦"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(url, proc):
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited early: {proc.stderr.read().decode()}")
        try:
            httpx.get(url, timeout=0.5)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"{url} never came up")


@pytest.fixture(scope="module")
def live():
    mock_port, gw_port = _free_port(), _free_port()
    env = {**os.environ, "GATEWAY_JWT_SECRET": SECRET, "PORT": str(gw_port),
           "DOWNSTREAM_URL": f"http://127.0.0.1:{mock_port}/mcp"}
    procs = [
        subprocess.Popen([sys.executable, str(ROOT / "mock_server.py")], env={**os.environ, "PORT": str(mock_port)},
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE),
        subprocess.Popen([sys.executable, str(ROOT / "gateway.py")], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE),
    ]
    try:
        _wait_for(f"http://127.0.0.1:{mock_port}/_debug/calls", procs[0])
        _wait_for(f"http://127.0.0.1:{gw_port}/healthz", procs[1])
        yield f"http://127.0.0.1:{gw_port}/mcp", f"http://127.0.0.1:{mock_port}/_debug/calls"
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)


def test_live_viewer_session(live):
    url, calls_url = live
    h = auth()
    init = httpx.post(url, json=rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                                   "clientInfo": {"name": "e2e", "version": "0"}}), headers=h)
    h["Mcp-Session-Id"] = init.headers["mcp-session-id"]
    assert httpx.post(url, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=h).status_code == 202
    tools = httpx.post(url, json=rpc("tools/list", msg_id=2), headers=h).json()["result"]["tools"]
    assert any(t["name"] == "admin_reset_key" for t in tools)
    denied = httpx.post(url, json=call("admin_reset_key", {"customer_id": "C1"}, msg_id=3), headers=h).json()
    assert denied["error"]["code"] == -32001
    ok = httpx.post(url, json=call("get_order", {"order_id": "A1"}, msg_id=4), headers=h).json()
    assert ok["result"]["isError"] is False

    seen = httpx.get(calls_url).json()
    assert [c["message"].get("method") for c in seen][-4:] == [
        "initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert all(c["message"].get("params", {}).get("name") != "admin_reset_key" for c in seen)
    assert all("authorization" not in c["headers"] for c in seen)


def test_live_admin_and_sse_stream(live):
    url, _ = live
    h = {**auth("admin"), "Accept": "application/json, text/event-stream"}
    assert httpx.post(url, json=call("admin_delete_user", {"user_id": "u1"}), headers=h).json()["result"]
    with httpx.stream("POST", url, json=call("stream_status", msg_id=5), headers=h) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        data = [json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")]
    assert [d.get("method", "result") for d in data] == ["notifications/progress", "result"]
