# Tests for the model router: failover on 429/timeout/errors, the timeout race, rate limiting through the
# API, sanitized error payloads, the circuit breaker, and a live run with the real 3000 ms deadline.

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_provider
import router

KEY, OTHER = "tenant-key-a", "tenant-key-b"
LEAK_MARKERS = ["Traceback", "10.0.0.5", "sk-live", "/srv/llm", "org-77", "127.0.0.1", "primary-model",
                "backup-model", "engine.py", "RuntimeError"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Harness:
    def __init__(self, tmp_path, primary_down=False, **overrides):
        self.primary = mock_provider.create_app("primary")
        self.backup = mock_provider.create_app("backup")
        options = {"tenant_keys": {KEY, OTHER}, "db_path": str(tmp_path / "gateway.db"),
                   "primary_timeout": 0.3, "backup_timeout": 1.0, **overrides}
        if primary_down:
            options["primary_url"] = f"http://127.0.0.1:{free_port()}/v1/chat/completions"
        self.settings = router.Settings(**options)
        self.app = router.create_app(
            self.settings,
            None if primary_down else httpx.ASGITransport(app=self.primary),
            httpx.ASGITransport(app=self.backup),
        )

    async def __aenter__(self):
        self._lifespan = self.app.router.lifespan_context(self.app)
        await self._lifespan.__aenter__()
        transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
        self.client = httpx.AsyncClient(transport=transport, base_url="http://gateway")
        return self

    async def __aexit__(self, *exc):
        await self.client.aclose()
        await self._lifespan.__aexit__(*exc)

    @property
    def p(self):
        return self.primary.state.mock

    @property
    def b(self):
        return self.backup.state.mock

    async def ask(self, content="hello", key=KEY, **extra):
        body = {"model": "anything", "messages": [{"role": "user", "content": content}], **extra}
        return await self.client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})

    def used(self, key=KEY):
        return self.app.state.limiter.used(router.tenant_id(key))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def assert_gateway_error(r, status, code):
    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "message", "request_id"}
    assert body["error"]["type"] == code
    assert body["error"]["request_id"].startswith("req_")
    assert r.headers["x-request-id"] == body["error"]["request_id"]
    for marker in LEAK_MARKERS:
        assert marker not in r.text, marker


@pytest.fixture
async def gw(tmp_path):
    async with Harness(tmp_path) as h:
        yield h


@pytest.mark.anyio
async def test_primary_success(gw):
    r = await gw.ask()
    assert r.status_code == 200
    assert r.headers["x-gateway-route"] == "primary"
    assert r.json()["choices"][0]["message"]["content"] == "answer from primary"
    assert gw.p["models"] == ["primary-model"]
    assert gw.b["received"] == 0
    assert gw.used() == r.json()["usage"]["total_tokens"]


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["429", "500", "401", "garbage", "no_choices"])
async def test_fails_over_to_backup(gw, mode):
    gw.p["mode"] = mode
    r = await gw.ask()
    assert r.status_code == 200
    assert r.headers["x-gateway-route"] == "backup"
    assert r.json()["choices"][0]["message"]["content"] == "answer from backup"
    assert gw.p["received"] == 1
    assert gw.b["models"] == ["backup-model"]


@pytest.mark.anyio
async def test_timeout_fails_over_and_cancels_primary(gw):
    gw.p["delay_ms"] = 2_000
    start = time.perf_counter()
    r = await gw.ask()
    elapsed = time.perf_counter() - start
    assert r.headers["x-gateway-route"] == "backup"
    assert 0.3 <= elapsed < 0.8
    assert gw.p["cancelled"] == 1
    assert gw.p["inflight"] == 0
    assert gw.p["completed"] == 0


@pytest.mark.anyio
async def test_primary_unreachable_fails_over(tmp_path):
    async with Harness(tmp_path, primary_down=True) as h:
        r = await h.ask()
    assert r.headers["x-gateway-route"] == "backup"


@pytest.mark.anyio
async def test_client_error_from_primary_does_not_fail_over(gw):
    gw.p["mode"] = "400"
    r = await gw.ask()
    assert_gateway_error(r, 400, "invalid_request")
    assert gw.b["received"] == 0
    assert gw.used() == 0


@pytest.mark.anyio
@pytest.mark.parametrize("primary, backup", [("500", "429"), ("429", "500"), ("slow", "slow"), ("garbage", "401")])
async def test_all_providers_failing_returns_503_and_releases_tokens(gw, primary, backup):
    for state, mode in ((gw.p, primary), (gw.b, backup)):
        if mode == "slow":
            state["delay_ms"] = 3_000
        else:
            state["mode"] = mode
    r = await gw.ask()
    assert_gateway_error(r, 503, "service_unavailable")
    assert r.headers["retry-after"] == "5"
    assert gw.used() == 0


@pytest.mark.anyio
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": f"Basic {KEY}"},
                                     {"Authorization": "Bearer"}])
async def test_bad_api_key(gw, headers):
    r = await gw.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]},
                             headers=headers)
    assert_gateway_error(r, 401, "authentication_error")
    assert gw.p["received"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        b"{not json",
        b"[1, 2]",
        b"{}",
        b'{"messages": []}',
        b'{"messages": ["hi"]}',
        b'{"messages": [{"content": "hi"}]}',
        b'{"messages": [{"role": "user", "content": 5}]}',
        b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}',
        b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": -5}',
        b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": "100"}',
        b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": true}',
        b'{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50001}',
        b'{"messages": [{"role": "user", "content": "hi"}], "stream": true}',
    ],
)
async def test_invalid_requests(gw, body):
    r = await gw.client.post("/v1/chat/completions", content=body,
                             headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    assert_gateway_error(r, 400, "invalid_request")
    assert gw.p["received"] == 0
    assert gw.used() == 0


@pytest.mark.anyio
async def test_reservation_is_corrected_to_actual_usage(gw):
    r = await gw.ask(max_tokens=4_000)
    actual = r.json()["usage"]["total_tokens"]
    assert actual < 100
    assert gw.used() == actual
    assert int(r.headers["x-ratelimit-remaining"]) == 50_000 - actual


@pytest.mark.anyio
async def test_rate_limit_through_the_api(tmp_path):
    async with Harness(tmp_path, tokens_per_minute=1_000) as h:
        h.p["completion_tokens"] = 300
        remaining = []
        for _ in range(3):
            r = await h.ask(max_tokens=300)
            assert r.status_code == 200
            remaining.append(int(r.headers["x-ratelimit-remaining"]))
        assert remaining == sorted(remaining, reverse=True)
        r = await h.ask(max_tokens=300)
        assert_gateway_error(r, 429, "rate_limit_exceeded")
        assert 1 <= int(r.headers["retry-after"]) <= 60
        assert h.p["received"] == 3
        assert (await h.ask(max_tokens=300, key=OTHER)).status_code == 200


@pytest.mark.anyio
async def test_request_bigger_than_limit(tmp_path):
    async with Harness(tmp_path, tokens_per_minute=1_000) as h:
        r = await h.ask(max_tokens=1_000)
    assert_gateway_error(r, 400, "request_too_large")


@pytest.mark.anyio
async def test_concurrent_requests_cannot_overspend(tmp_path):
    async with Harness(tmp_path, tokens_per_minute=1_000) as h:
        h.p["completion_tokens"] = 300
        h.p["delay_ms"] = 100
        responses = await asyncio.gather(*(h.ask(max_tokens=300) for _ in range(40)))
        ok = [r for r in responses if r.status_code == 200]
        limited = [r for r in responses if r.status_code == 429]
        assert len(ok) == 3
        assert len(limited) == 37
        assert h.used() <= 1_000


@pytest.mark.anyio
async def test_timeout_race_gives_one_answer_and_one_charge_per_request(gw):
    n = 30
    gw.p["delays_ms"] = [250 + (i * 100) // n for i in range(n)]
    responses = await asyncio.gather(*(gw.ask() for _ in range(n)))
    assert all(r.status_code == 200 for r in responses)
    routes = [r.headers["x-gateway-route"] for r in responses]
    for r in responses:
        assert r.json()["choices"][0]["message"]["content"] == f"answer from {r.headers['x-gateway-route']}"
    assert gw.p["completed"] == routes.count("primary")
    assert gw.p["completed"] + gw.p["cancelled"] == n
    assert gw.b["received"] == routes.count("backup")
    assert gw.p["inflight"] == 0
    assert 0 < routes.count("backup") < n
    assert gw.used() == sum(r.json()["usage"]["total_tokens"] for r in responses)
    assert gw.app.state.limiter.row_count() == n


@pytest.mark.anyio
async def test_circuit_breaker_skips_failing_primary_then_recovers(tmp_path):
    async with Harness(tmp_path, breaker_threshold=3, breaker_cooldown=0.5) as h:
        h.p["delay_ms"] = 2_000
        for _ in range(3):
            assert (await h.ask()).headers["x-gateway-route"] == "backup"
        start = time.perf_counter()
        r = await h.ask()
        assert r.headers["x-gateway-route"] == "backup"
        assert time.perf_counter() - start < 0.2
        assert h.p["received"] == 3

        await asyncio.sleep(0.6)
        r = await h.ask()
        assert r.headers["x-gateway-route"] == "backup"
        assert h.p["received"] == 4
        assert (await h.ask()).headers["x-gateway-route"] == "backup"
        assert h.p["received"] == 4

        await asyncio.sleep(0.6)
        h.p["delay_ms"] = 0
        assert (await h.ask()).headers["x-gateway-route"] == "primary"
        assert (await h.ask()).headers["x-gateway-route"] == "primary"


@pytest.mark.anyio
async def test_unknown_route_and_method(gw):
    assert_gateway_error(await gw.client.get("/nope"), 404, "not_found")
    assert_gateway_error(await gw.client.get("/v1/chat/completions"), 405, "method_not_allowed")


@pytest.mark.anyio
async def test_unexpected_crash_is_sanitized(gw, monkeypatch):
    def boom(_):
        raise RuntimeError("Traceback /srv/llm/engine.py sk-live-SECRET")

    monkeypatch.setattr(router, "estimate_prompt_tokens", boom)
    assert_gateway_error(await gw.ask(), 500, "internal_error")


@pytest.mark.anyio
async def test_crash_after_reservation_releases_tokens(gw, monkeypatch):
    async def boom(*args):
        raise RuntimeError("internal failure")

    monkeypatch.setattr(router, "route", boom)
    assert_gateway_error(await gw.ask(), 500, "internal_error")
    assert gw.used() == 0


def _wait_for_port(port, proc):
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
def live(tmp_path_factory):
    ports = {"primary": free_port(), "backup": free_port(), "router": free_port()}
    db = tmp_path_factory.mktemp("live") / "gateway.db"
    env = {
        **os.environ,
        "TENANT_API_KEYS": f"{KEY},{OTHER}",
        "DB_PATH": str(db),
        "PORT": str(ports["router"]),
        "PRIMARY_URL": f"http://127.0.0.1:{ports['primary']}/v1/chat/completions",
        "BACKUP_URL": f"http://127.0.0.1:{ports['backup']}/v1/chat/completions",
    }
    procs = [
        subprocess.Popen([sys.executable, str(ROOT / "mock_provider.py")],
                         env={**os.environ, "PORT": str(ports[name]), "MOCK_LABEL": name},
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for name in ("primary", "backup")
    ]
    procs.append(subprocess.Popen([sys.executable, str(ROOT / "router.py")], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    try:
        for name, proc in zip(("primary", "backup", "router"), procs):
            _wait_for_port(ports[name], proc)
        yield {name: f"http://127.0.0.1:{port}" for name, port in ports.items()}, db
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)


def live_ask(base):
    return httpx.post(f"{base}/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]},
                      headers={"Authorization": f"Bearer {KEY}"}, timeout=20)


def test_live_429_fails_over_immediately(live):
    urls, _ = live
    httpx.post(f"{urls['primary']}/_mock/behavior", json={"mode": "429", "delay_ms": 0})
    start = time.perf_counter()
    r = live_ask(urls["router"])
    assert r.headers["x-gateway-route"] == "backup"
    assert time.perf_counter() - start < 1.0


def test_live_3000ms_timeout_fails_over(live):
    urls, db = live
    httpx.post(f"{urls['primary']}/_mock/behavior", json={"mode": "ok", "delay_ms": 5_000})
    start = time.perf_counter()
    r = live_ask(urls["router"])
    elapsed = time.perf_counter() - start
    assert r.status_code == 200
    assert r.headers["x-gateway-route"] == "backup"
    assert 3.0 <= elapsed < 3.8
    assert db.exists() and db.stat().st_size > 0


def test_live_fast_primary_under_deadline(live):
    urls, _ = live
    httpx.post(f"{urls['primary']}/_mock/behavior", json={"mode": "ok", "delay_ms": 2_500})
    r = live_ask(urls["router"])
    assert r.headers["x-gateway-route"] == "primary"
