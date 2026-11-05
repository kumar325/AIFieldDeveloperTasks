# LLM gateway. Proxies OpenAI-style /v1/chat/completions requests to a provider and redacts PII from
# the response as it streams back, one SSE event at a time, without buffering the whole reply.

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from redactor import StreamRedactor, redact_text

log = logging.getLogger("llm-gateway")

MAX_REQUEST_BYTES = 2_000_000
STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def gateway_error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "gateway_error"}}, status_code=status)


async def sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data: list[str] = []
    async for line in lines:
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


def encode(payload: dict[str, Any] | str) -> bytes:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {body}\n\n".encode()


def is_empty_content_chunk(chunk: dict[str, Any]) -> bool:
    if set(chunk) - {"id", "object", "created", "model", "system_fingerprint", "choices"}:
        return False
    choices = chunk.get("choices") or []
    return bool(choices) and all(
        c.get("finish_reason") is None and set(c) <= {"index", "delta", "logprobs", "finish_reason"}
        and c.get("logprobs") is None and c.get("delta") == {"content": ""}
        for c in choices
    )


class StreamState:
    def __init__(self) -> None:
        self.redactors: dict[int, StreamRedactor] = {}
        self.envelope: dict[str, Any] = {}

    def redact_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        self.envelope = {k: v for k, v in chunk.items() if k in ("id", "object", "created", "model")}
        for choice in chunk.get("choices") or []:
            redactor = self.redactors.setdefault(choice.get("index", 0), StreamRedactor())
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                delta["content"] = redactor.feed(delta["content"])
            if choice.get("finish_reason") is not None:
                tail = redactor.flush()
                if tail or "content" in delta:
                    delta["content"] = delta.get("content", "") + tail
            choice["delta"] = delta
        return chunk

    def flush_all(self) -> dict[str, Any] | None:
        choices = []
        for index, redactor in self.redactors.items():
            tail = redactor.flush()
            if tail:
                choices.append({"index": index, "delta": {"content": tail}, "finish_reason": None})
        if not choices:
            return None
        return {**self.envelope, "choices": choices}


async def redact_stream(lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
    state = StreamState()
    async for data in sse_data(lines):
        if data == "[DONE]":
            leftover = state.flush_all()
            if leftover:
                yield encode(leftover)
            yield encode("[DONE]")
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            log.warning("dropping non-JSON SSE event from provider")
            continue
        if not isinstance(chunk, dict):
            continue
        chunk = state.redact_chunk(chunk)
        if not is_empty_content_chunk(chunk):
            yield encode(chunk)
    leftover = state.flush_all()
    if leftover:
        yield encode(leftover)


def redact_completion(body: dict[str, Any]) -> dict[str, Any]:
    for choice in body.get("choices") or []:
        message = choice.get("message") or {}
        if isinstance(message.get("content"), str):
            message["content"] = redact_text(message["content"])
    return body


async def chat_completions(request: Request) -> Response:
    raw = await request.body()
    if len(raw) > MAX_REQUEST_BYTES:
        return gateway_error(413, "Request body too large")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return gateway_error(400, "Request body must be valid JSON")
    if not isinstance(payload, dict):
        return gateway_error(400, "Request body must be a JSON object")

    app = request.app
    headers = {"Content-Type": "application/json"}
    if app.state.api_key:
        headers["Authorization"] = f"Bearer {app.state.api_key}"
    client: httpx.AsyncClient = app.state.client
    upstream_req = client.build_request("POST", app.state.provider_url, content=raw, headers=headers)

    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        return gateway_error(504, "Upstream provider timed out")
    except httpx.HTTPError as exc:
        log.warning("provider unreachable: %s", exc)
        return gateway_error(502, "Upstream provider unavailable")

    if upstream.status_code != 200:
        body = await upstream.aread()
        await upstream.aclose()
        return Response(body, status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type", "application/json"))

    if not payload.get("stream"):
        try:
            body = json.loads(await upstream.aread())
        finally:
            await upstream.aclose()
        return JSONResponse(redact_completion(body))

    async def body() -> AsyncIterator[bytes]:
        try:
            async for event in redact_stream(upstream.aiter_lines()):
                yield event
        except httpx.HTTPError as exc:
            log.warning("provider stream broke: %s", exc)
        finally:
            await upstream.aclose()

    return StreamingResponse(body(), media_type="text/event-stream", headers=STREAM_HEADERS,
                             background=BackgroundTask(upstream.aclose))


async def healthz(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


def create_app(provider_url: str, api_key: str = "", transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette):
        timeout = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)
        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            app.state.client = client
            yield

    app = Starlette(
        routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"]), Route("/healthz", healthz)],
        lifespan=lifespan,
    )
    app.state.provider_url = provider_url
    app.state.api_key = api_key
    return app


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    provider = os.environ.get("PROVIDER_URL", "http://127.0.0.1:9100/v1/chat/completions")
    port = int(os.environ.get("PORT", "8100"))
    log.info("gateway on http://127.0.0.1:%d/v1/chat/completions -> %s", port, provider)
    uvicorn.run(create_app(provider, os.environ.get("PROVIDER_API_KEY", "")),
                host=os.environ.get("HOST", "127.0.0.1"), port=port, log_level="warning")


if __name__ == "__main__":
    main()
