# Mock OpenAI-style LLM provider. Echoes the last user message back as the reply, streamed in small
# chunks with an optional delay, so tests can control exactly where PII gets split.

from __future__ import annotations

import asyncio
import json
import os
import time

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route


def chunk(cid: str, created: int, model: str, choices: list[dict]) -> bytes:
    body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": choices}
    return f"data: {json.dumps(body)}\n\n".encode()


async def completions(request: Request) -> Response:
    req = await request.json()
    if req.get("mock_status"):
        return JSONResponse({"error": {"message": "mock failure", "type": "mock"}}, status_code=req["mock_status"])

    messages = [m for m in req.get("messages", []) if m.get("role") == "user"]
    text = (messages[-1]["content"] if messages else "") * int(req.get("mock_repeat", 1))
    size = max(1, int(req.get("mock_chunk_size", 4)))
    delay = float(req.get("mock_delay_ms", 0)) / 1000
    n = int(req.get("n", 1))
    model = req.get("model", "mock-model")
    cid, created = "chatcmpl-mock", int(time.time())

    if not req.get("stream"):
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": i, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                        for i in range(n)],
        })

    async def stream():
        yield chunk(cid, created, model, [{"index": i, "delta": {"role": "assistant", "content": ""},
                                           "finish_reason": None} for i in range(n)])
        for start in range(0, len(text), size):
            if delay:
                await asyncio.sleep(delay)
            piece = text[start:start + size]
            for i in range(n):
                yield chunk(cid, created, model, [{"index": i, "delta": {"content": piece}, "finish_reason": None}])
        if req.get("mock_no_finish"):
            return
        yield chunk(cid, created, model, [{"index": i, "delta": {}, "finish_reason": "stop"} for i in range(n)])
        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def create_app() -> Starlette:
    return Starlette(routes=[Route("/v1/chat/completions", completions, methods=["POST"])])


if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "9100")), log_level="warning")
