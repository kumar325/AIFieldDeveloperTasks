# Tests for the LLM gateway: SSE format and redaction through the proxy, error handling, flat memory on
# huge streams, and real-HTTP checks that the stream stays incremental with a fast first token.

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gateway
import mock_provider
from redactor import REDACTED, redact_text

TEXT = "Contact jane@example.com, SSN 123-45-6789, card 4111 1111 1111 1111. Thanks for asking!"
EXPECTED = redact_text(TEXT)


@pytest.fixture
def client():
    app = gateway.create_app("http://provider/v1/chat/completions",
                             transport=httpx.ASGITransport(app=mock_provider.create_app()))
    with TestClient(app) as c:
        yield c


def request(text=TEXT, stream=True, **extra):
    return {"model": "mock", "stream": stream, "messages": [{"role": "user", "content": text}], **extra}


def events(body: str):
    out = []
    for block in body.split("\n\n"):
        if block.startswith("data: "):
            data = block[6:]
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


def content(evts, index=0):
    return "".join(
        c["delta"].get("content") or ""
        for e in evts if isinstance(e, dict)
        for c in e["choices"] if c["index"] == index
    )


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 16, 1000])
def test_stream_is_redacted_and_well_formed(client, chunk_size):
    r = client.post("/v1/chat/completions", json=request(mock_chunk_size=chunk_size))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    evts = events(r.text)
    assert evts[-1] == "[DONE]"
    assert content(evts) == EXPECTED
    for secret in ("jane@example.com", "6789", "1111 1111"):
        assert secret not in r.text
    assert evts[0]["choices"][0]["delta"]["role"] == "assistant"
    for e in evts[:-1]:
        assert e["object"] == "chat.completion.chunk" and e["id"] == "chatcmpl-mock"
    assert evts[-2]["choices"][0]["finish_reason"] == "stop"


def test_held_back_chunks_are_dropped_not_sent_empty(client):
    evts = events(client.post("/v1/chat/completions", json=request(mock_chunk_size=1)).text)
    for e in evts[1:-2]:
        assert e["choices"][0]["delta"]["content"] != ""


def test_multiple_choices_are_redacted_independently(client):
    evts = events(client.post("/v1/chat/completions", json=request(n=2, mock_chunk_size=3)).text)
    assert content(evts, 0) == EXPECTED
    assert content(evts, 1) == EXPECTED


def test_stream_without_finish_or_done_still_flushes(client):
    evts = events(client.post("/v1/chat/completions",
                              json=request("bye bob@example.com", mock_no_finish=True)).text)
    assert content(evts) == f"bye {REDACTED}"


def test_non_streaming_response_is_redacted(client):
    r = client.post("/v1/chat/completions", json=request(stream=False))
    assert r.json()["choices"][0]["message"]["content"] == EXPECTED


def test_provider_error_passes_through(client):
    r = client.post("/v1/chat/completions", json=request(mock_status=429))
    assert r.status_code == 429
    assert r.json()["error"]["message"] == "mock failure"


def test_provider_down_returns_502():
    with TestClient(gateway.create_app("http://127.0.0.1:1/v1/chat/completions")) as c:
        r = c.post("/v1/chat/completions", json=request())
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "gateway_error"


@pytest.mark.parametrize("body", [b"{bad json", b"[1, 2]", b'"text"'])
def test_bad_request_body(client, body):
    r = client.post("/v1/chat/completions", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_sse_parser_handles_multiline_and_comments():
    async def lines():
        for line in [": keepalive", "", "data: {\"a\":", "data: 1}", "", "event: x", "data:[DONE]", ""]:
            yield line

    async def collect():
        return [d async for d in gateway.sse_data(lines())]

    assert asyncio.run(collect()) == ['{"a":\n1}', "[DONE]"]


def test_memory_stays_flat_on_huge_stream():
    piece = 'data: {"id":"x","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"%s"},"finish_reason":null}]}'
    words = ["hello ", "world ", "jane@example.com ", "123-45-6789 ", "4111111111111111 "]
    total_chunks = 60_000

    async def lines():
        for i in range(total_chunks):
            yield piece % words[i % len(words)]
            yield ""
        yield "data: [DONE]"
        yield ""

    async def drain():
        sent = 0
        async for out in gateway.redact_stream(lines()):
            sent += len(out)
        return sent

    tracemalloc.start()
    sent = asyncio.run(drain())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert sent > 6_000_000
    assert peak < 2_000_000


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(port, proc):
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(proc.stderr.read().decode())
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"port {port} never opened")


@pytest.fixture(scope="module")
def live():
    provider_port, gw_port = _free_port(), _free_port()
    provider_url = f"http://127.0.0.1:{provider_port}/v1/chat/completions"
    procs = [
        subprocess.Popen([sys.executable, str(ROOT / "mock_provider.py")],
                         env={**os.environ, "PORT": str(provider_port)},
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE),
        subprocess.Popen([sys.executable, str(ROOT / "gateway.py")],
                         env={**os.environ, "PORT": str(gw_port), "PROVIDER_URL": provider_url},
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE),
    ]
    try:
        _wait_for(provider_port, procs[0])
        _wait_for(gw_port, procs[1])
        yield provider_url, f"http://127.0.0.1:{gw_port}/v1/chat/completions"
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)


def timed_stream(url, body):
    start = time.perf_counter()
    first_content = None
    arrivals = []
    text = []
    with httpx.stream("POST", url, json=body, timeout=30) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            piece = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if piece:
                now = time.perf_counter() - start
                first_content = first_content if first_content is not None else now
                arrivals.append(now)
                text.append(piece)
    return first_content, time.perf_counter() - start, arrivals, "".join(text)


def test_live_first_token_and_total_latency_close_to_provider(live):
    provider_url, gateway_url = live
    text = "Hello there, this is a streamed reply. " * 8 + "Email me at jane@example.com please."
    body = request(text, mock_chunk_size=4, mock_delay_ms=15)
    timed_stream(gateway_url, request("warm up"))

    direct_ttft, direct_total, _, _ = timed_stream(provider_url, body)
    gw_ttft, gw_total, arrivals, out = timed_stream(gateway_url, body)

    assert out == redact_text(text)
    assert gw_ttft < direct_ttft + 0.1
    assert gw_total < direct_total + 0.3
    assert len(arrivals) > 50
    assert arrivals[len(arrivals) // 2] < gw_total * 0.75


def test_live_client_disconnect_does_not_break_gateway(live):
    _, gateway_url = live
    body = request("word " * 2000, mock_chunk_size=2, mock_delay_ms=5)
    with httpx.stream("POST", gateway_url, json=body, timeout=30) as r:
        for i, _ in enumerate(r.iter_lines()):
            if i > 5:
                break
    r = httpx.post(gateway_url, json=request(stream=False), timeout=10)
    assert r.json()["choices"][0]["message"]["content"] == EXPECTED
