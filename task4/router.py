# Model router for an LLM gateway. Authenticates tenants by API key, enforces the per-tenant token limit,
# calls the primary model with a 3000 ms deadline, fails over to the backup, and returns sanitized errors.

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ratelimiter import RateLimited, RequestTooLarge, SlidingWindowLimiter

log = logging.getLogger("model-router")

MAX_REQUEST_BYTES = 1_000_000
NO_FAILOVER_STATUSES = {400, 413, 422}


@dataclass
class Settings:
    tenant_keys: set[str]
    db_path: str = "gateway.db"
    tokens_per_minute: int = 50_000
    primary_url: str = "http://127.0.0.1:9201/v1/chat/completions"
    primary_model: str = "primary-model"
    primary_api_key: str = ""
    primary_timeout: float = 3.0
    backup_url: str = "http://127.0.0.1:9202/v1/chat/completions"
    backup_model: str = "backup-model"
    backup_api_key: str = ""
    backup_timeout: float = 10.0
    default_max_tokens: int = 512
    breaker_threshold: int = 3
    breaker_cooldown: float = 10.0

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        keys = {k.strip() for k in env.get("TENANT_API_KEYS", "").split(",") if k.strip()}
        return cls(
            tenant_keys=keys,
            db_path=env.get("DB_PATH", "gateway.db"),
            tokens_per_minute=int(env.get("TOKENS_PER_MINUTE", "50000")),
            primary_url=env.get("PRIMARY_URL", cls.primary_url),
            primary_model=env.get("PRIMARY_MODEL", cls.primary_model),
            primary_api_key=env.get("PRIMARY_API_KEY", ""),
            primary_timeout=int(env.get("PRIMARY_TIMEOUT_MS", "3000")) / 1000,
            backup_url=env.get("BACKUP_URL", cls.backup_url),
            backup_model=env.get("BACKUP_MODEL", cls.backup_model),
            backup_api_key=env.get("BACKUP_API_KEY", ""),
            backup_timeout=int(env.get("BACKUP_TIMEOUT_MS", "10000")) / 1000,
        )


class ProviderError(Exception):
    def __init__(self, kind: str, status: int | None = None):
        super().__init__(kind if status is None else f"{kind} ({status})")
        self.kind = kind
        self.status = status

    @property
    def can_fail_over(self) -> bool:
        return self.kind != "rejected"


class AllProvidersFailed(Exception):
    pass


class CircuitBreaker:
    def __init__(self, threshold: int, cooldown: float, clock=time.monotonic):
        self.threshold = threshold
        self.cooldown = cooldown
        self.clock = clock
        self.failures = 0
        self.open_until = 0.0
        self.half_open = False

    def allow(self) -> bool:
        if self.open_until and self.clock() >= self.open_until:
            self.open_until = 0.0
            self.half_open = True
        return not self.open_until

    def record_success(self) -> None:
        self.failures = 0
        self.half_open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.half_open or self.failures >= self.threshold:
            self.open_until = self.clock() + self.cooldown
            self.failures = 0
            self.half_open = False


@dataclass
class Provider:
    name: str
    url: str
    model: str
    api_key: str
    timeout: float
    transport: httpx.AsyncBaseTransport | None = None
    client: httpx.AsyncClient | None = field(default=None, repr=False)

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with anyio.fail_after(self.timeout):
                resp = await self.client.post(self.url, json={**payload, "model": self.model}, headers=headers)
        except TimeoutError:
            raise ProviderError("timeout") from None
        except httpx.HTTPError:
            raise ProviderError("unreachable") from None

        if resp.status_code == 429:
            raise ProviderError("rate_limited", 429)
        if resp.status_code in NO_FAILOVER_STATUSES:
            raise ProviderError("rejected", resp.status_code)
        if resp.status_code >= 500:
            raise ProviderError("server_error", resp.status_code)
        if resp.status_code != 200:
            raise ProviderError("bad_status", resp.status_code)
        try:
            body = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ProviderError("bad_response") from None
        if not isinstance(body, dict) or not isinstance(body.get("choices"), list):
            raise ProviderError("bad_response")
        return body


def gateway_error(status: int, code: str, message: str, request_id: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": code, "message": message, "request_id": request_id}},
        status_code=status,
        headers={"X-Request-Id": request_id, **(headers or {})},
    )


def request_id(request: Request) -> str:
    if not hasattr(request.state, "request_id"):
        request.state.request_id = "req_" + secrets.token_hex(8)
    return request.state.request_id


def tenant_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def text_length(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(p.get("text", "")) for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return 0


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    return math.ceil(sum(text_length(m.get("content")) for m in messages) / 4) + 4 * len(messages) + 3


def validate(payload: Any, limit: int) -> str | None:
    if not isinstance(payload, dict):
        return "Request body must be a JSON object"
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return "'messages' must be a non-empty array"
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("role"), str):
            return "Each message needs a string 'role'"
        if not isinstance(m.get("content"), (str, list, type(None))):
            return "Message 'content' must be a string, array, or null"
    max_tokens = payload.get("max_tokens")
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                                   or not 1 <= max_tokens <= limit):
        return f"'max_tokens' must be an integer between 1 and {limit}"
    if payload.get("stream"):
        return "Streaming is not supported by this endpoint"
    return None


def actual_tokens(body: dict[str, Any], prompt_estimate: int) -> int:
    usage = body.get("usage")
    total = usage.get("total_tokens") if isinstance(usage, dict) else None
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        return total
    completion = sum(text_length((c.get("message") or {}).get("content"))
                     for c in body["choices"] if isinstance(c, dict))
    return prompt_estimate + math.ceil(completion / 4)


async def route(app: Starlette, payload: dict[str, Any], rid: str) -> tuple[str, dict[str, Any]]:
    primary: Provider = app.state.primary
    backup: Provider = app.state.backup
    breaker: CircuitBreaker = app.state.breaker

    if breaker.allow():
        try:
            body = await primary.complete(payload)
            breaker.record_success()
            return primary.name, body
        except ProviderError as exc:
            log.warning("%s primary failed: %s", rid, exc)
            if not exc.can_fail_over:
                raise
            breaker.record_failure()
    else:
        log.info("%s primary circuit open, using backup", rid)

    try:
        return backup.name, await backup.complete(payload)
    except ProviderError as exc:
        log.error("%s backup failed: %s", rid, exc)
        if not exc.can_fail_over:
            raise
        raise AllProvidersFailed from None


async def chat_completions(request: Request) -> Response:
    rid = request_id(request)
    app = request.app
    settings: Settings = app.state.settings
    limiter: SlidingWindowLimiter = app.state.limiter

    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    key = key.strip()
    if scheme.lower() != "bearer" or key not in settings.tenant_keys:
        return gateway_error(401, "authentication_error", "Missing or invalid API key", rid,
                             {"WWW-Authenticate": "Bearer"})
    tenant = tenant_id(key)

    raw = await request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        return gateway_error(413, "request_too_large", "Request body is too large", rid)
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return gateway_error(400, "invalid_request", "Request body must be valid JSON", rid)
    problem = validate(payload, settings.tokens_per_minute)
    if problem:
        return gateway_error(400, "invalid_request", problem, rid)

    prompt_estimate = estimate_prompt_tokens(payload["messages"])
    estimate = prompt_estimate + (payload.get("max_tokens") or settings.default_max_tokens)
    try:
        reservation = await asyncio.to_thread(limiter.reserve, tenant, estimate)
    except RequestTooLarge:
        return gateway_error(400, "request_too_large",
                             "This request needs more tokens than the per-minute limit allows", rid)
    except RateLimited as exc:
        retry = max(1, math.ceil(exc.retry_after))
        return gateway_error(429, "rate_limit_exceeded", f"Token rate limit reached. Retry in {retry}s.", rid,
                             {"Retry-After": str(retry), "X-RateLimit-Limit": str(exc.limit),
                              "X-RateLimit-Remaining": str(max(0, exc.limit - exc.used))})

    try:
        provider_name, body = await route(app, payload, rid)
    except BaseException as exc:
        with anyio.CancelScope(shield=True):
            await asyncio.to_thread(limiter.release, reservation)
        if isinstance(exc, ProviderError):
            return gateway_error(400, "invalid_request", "The model rejected this request", rid)
        if isinstance(exc, AllProvidersFailed):
            return gateway_error(503, "service_unavailable",
                                 "All model providers are unavailable. Please retry shortly.", rid,
                                 {"Retry-After": "5"})
        raise

    used = actual_tokens(body, prompt_estimate)
    remaining = await asyncio.to_thread(limiter.commit, reservation, used)
    log.info("%s tenant=%s route=%s tokens=%d", rid, tenant, provider_name, used)
    return JSONResponse(body, headers={
        "X-Request-Id": rid,
        "X-Gateway-Route": provider_name,
        "X-RateLimit-Limit": str(settings.tokens_per_minute),
        "X-RateLimit-Remaining": str(max(0, remaining)),
    })


async def healthz(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def http_error(request: Request, exc: HTTPException) -> Response:
    messages = {404: ("not_found", "Not found"), 405: ("method_not_allowed", "Method not allowed")}
    code, message = messages.get(exc.status_code, ("invalid_request", "Bad request"))
    return gateway_error(exc.status_code, code, message, request_id(request))


async def unexpected_error(request: Request, exc: Exception) -> Response:
    rid = request_id(request)
    log.error("%s unhandled error", rid, exc_info=exc)
    return gateway_error(500, "internal_error", "Something went wrong on our side", rid)


def create_app(
    settings: Settings,
    primary_transport: httpx.AsyncBaseTransport | None = None,
    backup_transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    primary = Provider("primary", settings.primary_url, settings.primary_model, settings.primary_api_key,
                       settings.primary_timeout, primary_transport)
    backup = Provider("backup", settings.backup_url, settings.backup_model, settings.backup_api_key,
                      settings.backup_timeout, backup_transport)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        async with httpx.AsyncClient(transport=primary.transport, limits=limits,
                                     timeout=settings.primary_timeout + 5) as p, \
                httpx.AsyncClient(transport=backup.transport, limits=limits,
                                  timeout=settings.backup_timeout + 5) as b:
            primary.client, backup.client = p, b
            yield

    app = Starlette(
        routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"]), Route("/healthz", healthz)],
        exception_handlers={HTTPException: http_error, Exception: unexpected_error},
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.limiter = SlidingWindowLimiter(settings.db_path, settings.tokens_per_minute)
    app.state.primary = primary
    app.state.backup = backup
    app.state.breaker = CircuitBreaker(settings.breaker_threshold, settings.breaker_cooldown)
    return app


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings.from_env()
    if not settings.tenant_keys:
        sys.exit("TENANT_API_KEYS must list at least one API key")
    port = int(os.environ.get("PORT", "8200"))
    log.info("router on http://127.0.0.1:%d/v1/chat/completions (primary timeout %.0f ms)",
             port, settings.primary_timeout * 1000)
    uvicorn.run(create_app(settings), host=os.environ.get("HOST", "127.0.0.1"), port=port, log_level="warning")


if __name__ == "__main__":
    main()
