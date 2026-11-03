"""Customer support MCP server over stdio."""

from __future__ import annotations

# Runs before the other imports so nothing printed at import time hits stdout.
import os
import sys


def _isolate_stdout() -> int:
    """Point fd 1 at stderr and return a private fd for the real stdout."""
    sys.stdout.flush()
    wire_fd = os.dup(1)
    os.dup2(2, 1)
    if sys.platform == "win32":
        # dup2 doesn't update the Win32 handle that child processes inherit.
        import ctypes
        import msvcrt

        STD_OUTPUT_HANDLE = -11
        ctypes.windll.kernel32.SetStdHandle(STD_OUTPUT_HANDLE, msvcrt.get_osfhandle(1))
    # Windows pipes default to cp1252.
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stdout = sys.stderr
    return wire_fd


_WIRE_FD = _isolate_stdout()

import json  # noqa: E402
import logging  # noqa: E402
from decimal import Decimal  # noqa: E402
from typing import Annotated, Any  # noqa: E402

import anyio  # noqa: E402
import mcp_types as types  # noqa: E402
from mcp.server.lowlevel import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.shared._context_streams import create_context_streams  # noqa: E402
from mcp.shared.exceptions import MCPError  # noqa: E402
from mcp.shared.message import SessionMessage  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator  # noqa: E402

logging.basicConfig(
    stream=sys.stderr,
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("customer-mcp")

# [0-9], not \d, since \d also matches non-ASCII digits.
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
MAX_REFUND = 10_000.00

CustomerId = Annotated[
    str,
    Field(
        pattern=CUSTOMER_ID_PATTERN,
        description="Customer ID in the form CUST-XXXXX, where X is a digit 0-9.",
        examples=["CUST-00042"],
    ),
]


class StrictInput(BaseModel):
    # strict: no type coercion. forbid: reject unknown fields.
    model_config = ConfigDict(strict=True, extra="forbid")


class GetCustomerRecordInput(StrictInput):
    customer_id: CustomerId


class TriggerRefundInput(StrictInput):
    customer_id: CustomerId
    amount: Annotated[
        float,
        Field(
            gt=0,
            le=MAX_REFUND,
            allow_inf_nan=False,
            description=f"Refund amount in USD. Must be > 0, at most {MAX_REFUND:.2f}, max 2 decimal places.",
        ),
    ]
    reason: Annotated[
        str,
        Field(
            min_length=10,
            max_length=500,
            description="Why the refund is issued. At least 10 non-whitespace-padded characters.",
        ),
    ]

    @field_validator("amount")
    @classmethod
    def at_most_two_decimals(cls, v: float) -> float:
        if Decimal(str(v)).as_tuple().exponent < -2:
            raise ValueError("amount must have at most 2 decimal places")
        return v

    @field_validator("reason")
    @classmethod
    def meaningful_reason(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("reason must contain at least 10 characters excluding leading/trailing whitespace")
        if any(ord(c) < 32 and c not in "\t\n\r" for c in v):
            raise ValueError("reason must not contain control characters")
        return v


TOOLS: dict[str, tuple[type[StrictInput], str]] = {
    "get_customer_record": (
        GetCustomerRecordInput,
        "Look up a customer's record by ID.",
    ),
    "trigger_refund": (
        TriggerRefundInput,
        "Issue a refund to a customer. Fails if the amount exceeds the customer's refundable balance.",
    ),
}

CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"name": "Ada Lovelace", "email": "ada@example.com", "tier": "gold", "refundable_balance": 500.00},
    "CUST-12345": {"name": "Alan Turing", "email": "alan@example.com", "tier": "silver", "refundable_balance": 120.50},
    "CUST-99999": {"name": "Grace Hopper", "email": "grace@example.com", "tier": "bronze", "refundable_balance": 0.00},
}
REFUNDS: list[dict[str, Any]] = []


class BusinessError(Exception):
    """Valid input the business logic can't fulfil. Sent back as isError."""


def get_customer_record(args: GetCustomerRecordInput) -> dict[str, Any]:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        raise BusinessError(f"Customer {args.customer_id} not found")
    return {"customer_id": args.customer_id, **record}


def trigger_refund(args: TriggerRefundInput) -> dict[str, Any]:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        raise BusinessError(f"Customer {args.customer_id} not found")
    balance = Decimal(str(record["refundable_balance"]))
    amount = Decimal(str(args.amount))
    if amount > balance:
        raise BusinessError(
            f"Refund of {amount:.2f} exceeds refundable balance of {balance:.2f} for {args.customer_id}"
        )
    record["refundable_balance"] = float(balance - amount)
    refund = {
        "refund_id": f"RF-{len(REFUNDS) + 1:06d}",
        "customer_id": args.customer_id,
        "amount": float(amount),
        "reason": args.reason.strip(),
        "status": "issued",
        "remaining_refundable_balance": record["refundable_balance"],
    }
    REFUNDS.append(refund)
    return refund


HANDLERS = {
    "get_customer_record": get_customer_record,
    "trigger_refund": trigger_refund,
}


def _input_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(name=name, description=desc, input_schema=_input_schema(model))
            for name, (model, desc) in TOOLS.items()
        ]
    )


def _validation_error(tool: str, exc: ValidationError) -> MCPError:
    errors = [
        {
            "field": ".".join(str(p) for p in e["loc"]) or "(root)",
            "message": e["msg"],
            "type": e["type"],
        }
        for e in exc.errors(include_url=False, include_input=False)
    ]
    summary = "; ".join(f"{e['field']}: {e['message']}" for e in errors)
    return MCPError(
        code=types.INVALID_PARAMS,
        message=f"Invalid arguments for {tool}: {summary}",
        data={"tool": tool, "errors": errors},
    )


async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    name = params.name
    if name not in TOOLS:
        log.warning("call to unknown tool %r", name)
        raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {name}", data={"tool": name})

    model, _ = TOOLS[name]
    raw_args = params.arguments if params.arguments is not None else {}
    try:
        args = model.model_validate(raw_args)
    except ValidationError as exc:
        err = _validation_error(name, exc)
        log.info("rejected %s: %s", name, err.message)
        raise err from None

    try:
        result = HANDLERS[name](args)
    except BusinessError as exc:
        log.info("%s failed: %s", name, exc)
        return types.CallToolResult(content=[types.TextContent(type="text", text=str(exc))], is_error=True)
    except Exception:
        # The SDK would send this as code 0 on stdio, so map it to -32603 here.
        log.exception("unexpected error in %s", name)
        raise MCPError(code=types.INTERNAL_ERROR, message="Internal server error") from None

    log.info("%s ok", name)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(result))],
        structured_content=result,
    )


server = Server(
    "customer-support",
    version="1.0.0",
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
)


def _transport_error(exc: Exception) -> types.JSONRPCError:
    is_bad_json = isinstance(exc, ValidationError) and any(e["type"] == "json_invalid" for e in exc.errors())
    if is_bad_json:
        error = types.ErrorData(code=types.PARSE_ERROR, message="Parse error: message is not valid JSON")
    else:
        error = types.ErrorData(code=types.INVALID_REQUEST, message="Invalid Request: not a valid JSON-RPC 2.0 message")
    return types.JSONRPCError(jsonrpc="2.0", id=None, error=error)


async def _answer_unparseable(read_stream: Any, write_stream: Any, forward: Any) -> None:
    """Reply to unparseable lines with id=null errors. The SDK drops them silently."""
    async with forward:
        async for item in read_stream:
            if isinstance(item, Exception):
                reply = _transport_error(item)
                log.warning("unparseable inbound line -> %s", reply.error.code)
                await write_stream.send(SessionMessage(reply))
                continue
            await forward.send(item)


async def main() -> None:
    wire = anyio.wrap_file(open(_WIRE_FD, "w", encoding="utf-8", newline="\n", closefd=False))
    log.info("customer-support MCP server starting on stdio")
    async with stdio_server(stdout=wire) as (read_stream, write_stream):
        forward, filtered = create_context_streams[SessionMessage | Exception](0)
        async with anyio.create_task_group() as tg:
            tg.start_soon(_answer_unparseable, read_stream, write_stream, forward)
            await server.run(filtered, write_stream, server.create_initialization_options())
            tg.cancel_scope.cancel()
    log.info("stdin closed, shutting down")


if __name__ == "__main__":
    anyio.run(main)
