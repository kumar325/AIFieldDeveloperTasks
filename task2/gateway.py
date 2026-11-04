"""MCP security gateway: bearer auth and tool-level authorization in front of an MCP server."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Any

import httpx
import jwt
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

log = logging.getLogger("mcp-gateway")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHENTICATED = -32000
UNAUTHORIZED_TOOL = -32001

ROLES = {"admin", "viewer"}
MAX_BODY_BYTES = 1_000_000
# Charset from the MCP tool naming guidance. Rejecting everything else blocks
# lookalike tricks like "ａｄｍｉｎ_x" or a zero-width space before "admin_".
TOOL_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,128}")

FORWARD_REQUEST_HEADERS = {"accept", "content-type", "mcp-session-id", "mcp-protocol-version", "last-event-id"}
FORWARD_RESPONSE_HEADERS = {"content-type", "mcp-session-id", "mcp-protocol-version", "cache-control"}


class GatewayError(Exception):
    def __init__(self, code: int, message: str, status: int = 200, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.data = data


class DuplicateKeyError(ValueError):
    pass


def rpc_error(
    msg_id: Any, code: int, message: str, status: int = 200, data: Any = None, headers: dict | None = None
) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": msg_id, "error": error}, status_code=status, headers=headers)


def authenticate(header: str | None, secret: str) -> dict[str, Any]:
    if not header:
        raise GatewayError(UNAUTHENTICATED, "Missing bearer token", 401)
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise GatewayError(UNAUTHENTICATED, "Authorization header must be 'Bearer <token>'", 401)
    try:
        # Pinning the algorithm rules out alg=none and key-confusion tokens.
        claims = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "sub", "role"]})
    except jwt.ExpiredSignatureError:
        raise GatewayError(UNAUTHENTICATED, "Token expired", 401) from None
    except jwt.PyJWTError:
        raise GatewayError(UNAUTHENTICATED, "Invalid token", 401) from None
    role = claims.get("role")
    if not isinstance(role, str) or role not in ROLES:
        raise GatewayError(UNAUTHENTICATED, "Token has no valid role", 401)
    return claims


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # Duplicate keys let a client show us one tool name and the server another.
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise DuplicateKeyError(key)
        obj[key] = value
    return obj


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not valid JSON")


def _valid_id(value: Any) -> bool:
    return value is None or isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def parse_message(body: bytes) -> dict[str, Any]:
    try:
        msg = json.loads(body, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except DuplicateKeyError as exc:
        raise GatewayError(INVALID_REQUEST, f"Duplicate JSON key: {exc}", 400) from None
    except (ValueError, UnicodeDecodeError):
        raise GatewayError(PARSE_ERROR, "Parse error", 400) from None

    if isinstance(msg, list):
        # MCP dropped JSON-RPC batching. Refusing it also means one array can't hide an admin call.
        raise GatewayError(INVALID_REQUEST, "Batch requests are not supported", 400)
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        raise GatewayError(INVALID_REQUEST, "Invalid Request", 400)
    if "id" in msg and not _valid_id(msg["id"]):
        raise GatewayError(INVALID_REQUEST, "id must be a string, integer, or null", 400)

    if "method" in msg:
        if not isinstance(msg["method"], str) or "result" in msg or "error" in msg:
            raise GatewayError(INVALID_REQUEST, "Invalid Request", 400)
        if "params" in msg and not isinstance(msg["params"], (dict, list)):
            raise GatewayError(INVALID_REQUEST, "params must be an object or array", 400)
    elif "id" not in msg or ("result" in msg) == ("error" in msg):
        # Not a request, so it has to be a response with exactly one of result/error.
        raise GatewayError(INVALID_REQUEST, "Invalid Request", 400)
    return msg


def is_admin_tool(name: str) -> bool:
    # Case-insensitive so "ADMIN_reset_key" can't slip past a case-sensitive check.
    return name.lower().startswith("admin_")


def authorize(msg: dict[str, Any], role: str) -> None:
    if msg.get("method") != "tools/call":
        return
    if "id" not in msg:
        raise GatewayError(INVALID_REQUEST, "tools/call must be a request with an id", 400)
    params = msg.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str):
        raise GatewayError(INVALID_PARAMS, "tools/call requires params.name as a string")
    if not TOOL_NAME.fullmatch(name):
        raise GatewayError(INVALID_PARAMS, "Invalid tool name", data={"tool": name})
    if is_admin_tool(name) and role != "admin":
        raise GatewayError(UNAUTHORIZED_TOOL, "Unauthorized Tool Call", data={"tool": name, "required_role": "admin"})


async def forward(request: Request, body: bytes | None, msg_id: Any = None) -> Response:
    client: httpx.AsyncClient = request.app.state.client
    headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD_REQUEST_HEADERS}
    # GET opens a long-lived SSE stream, so it gets no read timeout.
    timeout = httpx.Timeout(30.0, connect=5.0, read=None if request.method == "GET" else 30.0)
    upstream_req = client.build_request(
        request.method, request.app.state.downstream_url, headers=headers, content=body, timeout=timeout
    )
    try:
        upstream = await client.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        log.warning("downstream timed out")
        return rpc_error(msg_id, INTERNAL_ERROR, "Downstream server timed out", 504)
    except httpx.HTTPError as exc:
        log.warning("downstream unreachable: %s", exc)
        return rpc_error(msg_id, INTERNAL_ERROR, "Downstream server unavailable", 502)

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() in FORWARD_RESPONSE_HEADERS}
    # Streamed so SSE responses pass through event by event.
    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        headers=resp_headers,
        background=BackgroundTask(upstream.aclose),
    )


async def read_body(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise GatewayError(INVALID_REQUEST, "Request body too large", 413)
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise GatewayError(INVALID_REQUEST, "Request body too large", 413)
    return bytes(body)


async def mcp_endpoint(request: Request) -> Response:
    msg_id = None
    try:
        claims = authenticate(request.headers.get("authorization"), request.app.state.secret)
        role, sub = claims["role"], claims["sub"]

        if request.method != "POST":
            log.info("allow %s sub=%s role=%s", request.method, sub, role)
            return await forward(request, None)

        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise GatewayError(INVALID_REQUEST, "Content-Type must be application/json", 415)

        msg = parse_message(await read_body(request))
        msg_id = msg.get("id")
        method = msg.get("method", "<response>")
        try:
            authorize(msg, role)
        except GatewayError as exc:
            log.info("deny %s sub=%s role=%s code=%s data=%s", method, sub, role, exc.code, exc.data)
            raise
        log.info("allow %s sub=%s role=%s", method, sub, role)
        # Forward the message we checked, re-serialized, not the raw bytes.
        return await forward(request, json.dumps(msg, separators=(",", ":")).encode(), msg_id)

    except GatewayError as exc:
        headers = {"WWW-Authenticate": 'Bearer error="invalid_token"'} if exc.status == 401 else None
        if exc.status == 401:
            log.info("reject: %s", exc.message)
        return rpc_error(msg_id, exc.code, exc.message, exc.status, exc.data, headers)


async def healthz(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


def create_app(downstream_url: str, secret: str, transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            app.state.client = client
            yield

    app = Starlette(
        routes=[Route("/mcp", mcp_endpoint, methods=["POST", "GET", "DELETE"]), Route("/healthz", healthz)],
        lifespan=lifespan,
    )
    app.state.downstream_url = downstream_url
    app.state.secret = secret
    return app


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    secret = os.environ.get("GATEWAY_JWT_SECRET", "")
    if len(secret) < 32:
        sys.exit("GATEWAY_JWT_SECRET must be set to at least 32 characters")
    downstream = os.environ.get("DOWNSTREAM_URL", "http://127.0.0.1:9000/mcp")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    log.info("gateway on http://%s:%d/mcp -> %s", host, port, downstream)
    uvicorn.run(create_app(downstream, secret), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
