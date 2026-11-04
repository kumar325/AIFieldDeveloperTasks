"""Mock downstream MCP server (Streamable HTTP, JSON and SSE responses)."""

from __future__ import annotations

import json
import os
import uuid

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

TOOLS = [
    {"name": "get_order", "description": "Look up an order.",
     "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "stream_status", "description": "Streams progress, then a result (SSE).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "admin_reset_key", "description": "Rotate a customer's API key.",
     "inputSchema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]}},
    {"name": "admin_delete_user", "description": "Delete a user account.",
     "inputSchema": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]}},
]
TOOL_NAMES = {t["name"] for t in TOOLS}


def result(msg_id, payload):
    return {"jsonrpc": "2.0", "id": msg_id, "result": payload}


def text(msg_id, s):
    return result(msg_id, {"content": [{"type": "text", "text": s}], "isError": False})


def sse(*events):
    async def gen():
        for event in events:
            yield f"event: message\ndata: {json.dumps(event)}\n\n".encode()
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


async def mcp(request: Request) -> Response:
    app = request.app
    if request.method == "GET":
        return Response(status_code=405)
    if request.method == "DELETE":
        return Response(status_code=200)

    msg = await request.json()
    app.state.calls.append({"message": msg, "headers": dict(request.headers)})
    method, msg_id = msg.get("method"), msg.get("id")

    if method is None:  # a response from the client
        return Response(status_code=202)
    if method.startswith("notifications/"):
        return Response(status_code=202)
    if method == "initialize":
        body = result(msg_id, {
            "protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-11-25"),
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "mock-downstream", "version": "1.0.0"},
        })
        return JSONResponse(body, headers={"Mcp-Session-Id": uuid.uuid4().hex})
    if method == "ping":
        return JSONResponse(result(msg_id, {}))
    if method == "tools/list":
        return JSONResponse(result(msg_id, {"tools": TOOLS}))
    if method == "tools/call":
        name = msg["params"]["name"]
        args = msg["params"].get("arguments") or {}
        if name not in TOOL_NAMES:
            return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32602, "message": f"Unknown tool: {name}"}})
        if name == "stream_status":
            progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progressToken": 1, "progress": 50}}
            return sse(progress, text(msg_id, "done"))
        if name == "get_order":
            return JSONResponse(text(msg_id, json.dumps({"order_id": args.get("order_id"), "status": "shipped"})))
        if name == "admin_reset_key":
            return JSONResponse(text(msg_id, f"key rotated for {args.get('customer_id')}"))
        return JSONResponse(text(msg_id, f"deleted {args.get('user_id')}"))
    return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": "Method not found"}})


async def debug_calls(request: Request) -> Response:
    # Lets end-to-end tests check what reached the server.
    return JSONResponse(request.app.state.calls)


def create_app() -> Starlette:
    app = Starlette(routes=[
        Route("/mcp", mcp, methods=["POST", "GET", "DELETE"]),
        Route("/_debug/calls", debug_calls),
    ])
    app.state.calls = []
    return app


if __name__ == "__main__":
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("PORT", "9000")), log_level="warning")
