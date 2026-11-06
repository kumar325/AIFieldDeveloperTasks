# Mock OpenAI-style provider used as the primary or backup model in tests. Its behaviour (ok, slow, 429,
# 500, bad output) can be switched at runtime, and it counts in-flight and cancelled requests.

from __future__ import annotations

import asyncio
import math
import os
import time

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

LEAKY_DETAIL = "Traceback (most recent call last): File /srv/llm/engine.py line 88 host=10.0.0.5 key=sk-live-SECRET"


def default_state(label: str) -> dict:
    return {"label": label, "mode": "ok", "delay_ms": 0, "delays_ms": [], "completion_tokens": 12,
            "received": 0, "completed": 0, "cancelled": 0, "inflight": 0, "models": []}


async def completions(request: Request) -> Response:
    s = request.app.state.mock
    body = await request.json()
    s["received"] += 1
    s["inflight"] += 1
    s["models"].append(body.get("model"))
    try:
        delay = s["delays_ms"].pop(0) if s["delays_ms"] else s["delay_ms"]
        if delay:
            await asyncio.sleep(delay / 1000)
        mode = s["mode"]
        if mode == "429":
            return JSONResponse({"error": {"message": f"Rate limit for org-77 on {LEAKY_DETAIL}"}},
                                status_code=429, headers={"Retry-After": "20"})
        if mode == "500":
            return PlainTextResponse(LEAKY_DETAIL, status_code=500)
        if mode in ("400", "401"):
            return JSONResponse({"error": {"message": LEAKY_DETAIL}}, status_code=int(mode))
        if mode == "garbage":
            return PlainTextResponse("<html>upstream proxy error</html>")
        if mode == "no_choices":
            return JSONResponse({"id": "x", "object": "chat.completion"})
        prompt = math.ceil(sum(len(m.get("content") or "") for m in body.get("messages", [])) / 4)
        completion = s["completion_tokens"]
        s["completed"] += 1
        return JSONResponse({
            "id": f"chatcmpl-{s['label']}", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": f"answer from {s['label']}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
        })
    except asyncio.CancelledError:
        s["cancelled"] += 1
        raise
    finally:
        s["inflight"] -= 1


async def set_behavior(request: Request) -> Response:
    request.app.state.mock.update(await request.json())
    return JSONResponse(request.app.state.mock)


async def stats(request: Request) -> Response:
    return JSONResponse(request.app.state.mock)


def create_app(label: str = "mock") -> Starlette:
    app = Starlette(routes=[
        Route("/v1/chat/completions", completions, methods=["POST"]),
        Route("/_mock/behavior", set_behavior, methods=["POST"]),
        Route("/_mock/stats", stats),
    ])
    app.state.mock = default_state(label)
    return app


if __name__ == "__main__":
    app = create_app(os.environ.get("MOCK_LABEL", "mock"))
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "9201")), log_level="warning")
