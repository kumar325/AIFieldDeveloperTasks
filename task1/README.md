# Customer Support MCP Server

A stdio MCP server built on the official Python SDK (`mcp` 2.3, low-level `Server`) with Pydantic v2 validation. It exposes two tools:

| Tool | Inputs |
|---|---|
| `get_customer_record` | `customer_id`: `CUST-XXXXX` |
| `trigger_refund` | `customer_id`, `amount` (positive number), `reason` (≥ 10 chars) |

## Run it

From the `task1/` folder:

```bash
pip install -r requirements.txt
python server.py            # speaks JSON-RPC on stdin/stdout
python -m pytest -q         # end-to-end tests over real stdio
```

Inspect it interactively with `npx @modelcontextprotocol/inspector python server.py`.

To use it from an MCP client such as Claude Desktop:

```json
{
  "mcpServers": {
    "customer-support": { "command": "python", "args": ["/absolute/path/to/server.py"] }
  }
}
```

Set `LOG_LEVEL=DEBUG` in the environment for more detail on stderr.

### Windows

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -v
```

In a client config on Windows, use full paths with escaped backslashes:

```json
"command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
"args": ["C:\\path\\to\\server.py"]
```

Windows-specific handling in the server:
- The stdout guard repoints the Win32 `STD_OUTPUT_HANDLE` as well as fd 1, because child processes inherit the handle, not the CRT descriptor.
- stderr is forced to UTF-8 so logs can't crash on non-ASCII text under cp1252.
- The tests assert that no frame ends in `\r\n`.

CI runs the suite on Windows, macOS and Linux (`.github/workflows/task1.yml` at the repo root).

## STDIO isolation

stdout is reserved for JSON-RPC frames. The server enforces this at the file-descriptor level, not by discipline alone.

1. **Before any other import**, `_isolate_stdout()` duplicates fd 1 onto a private descriptor, then points fd 1 at stderr and sets `sys.stdout = sys.stderr`.
2. The private descriptor is handed to `stdio_server(stdout=...)` explicitly. It's the only path to the client.
3. All logging goes through `logging` configured with `stream=sys.stderr`.

The result is that `print()`, `sys.stdout.write()`, raw `os.write(1, ...)`, C extensions, import-time banners from dependencies, and child processes all land on stderr. None of them can corrupt the protocol stream. `test_stray_writes_after_guard_never_reach_stdout` checks all four write paths. `test_source_has_no_stdout_writes` statically checks the source with the AST.

(SDK 2.3 also diverts fd 1 while `stdio_server()` is running. The guard here covers the window before that, during imports, and doesn't depend on SDK internals.)

## Error mapping

There are two channels, chosen deliberately:

| Situation | Response |
|---|---|
| Line isn't valid JSON | JSON-RPC error `-32700`, `id: null` |
| Valid JSON but not a JSON-RPC 2.0 message (wrong shape, batch array, `"jsonrpc":"1.0"`) | `-32600`, `id: null` |
| Unknown method | `-32601` |
| Unknown tool name | `-32602` |
| Arguments fail schema validation | `-32602`, with `data.errors[]` listing `field` / `message` / `type` for every failure |
| Unexpected exception in a handler | `-32603 "Internal server error"`. The traceback goes to stderr and never reaches the wire. |
| Well-formed request the business logic can't fulfil (customer not found, refund exceeds balance) | Normal result with `isError: true` |

Notes on two of these:

- **The SDK drops unparseable lines silently.** JSON-RPC 2.0 requires an error with `id: null`, so a small filter (`_answer_unparseable`) sits between the transport and the server to send one.
- **Unhandled exceptions go out as `code: 0` on stdio in SDK 2.3** (a v1 compatibility choice). The tool handler catches everything and raises `MCPError(-32603)` so the client always gets a standard code.

**Validation errors as protocol errors vs `isError`.** Recent MCP spec revisions lean toward returning input validation failures as `isError` tool results, so a model can read the message and retry. This task asks for standard JSON-RPC error codes, so invalid input returns `-32602`. The structured `data.errors` payload keeps it just as self-correctable. Switching would be a single change in `on_call_tool`.

## Validation decisions

All inputs use `strict=True` (no type coercion) and `extra="forbid"` (advertised as `additionalProperties: false`).

**`customer_id`**
- Pattern `^CUST-[0-9]{5}$`. `X` is read as a decimal digit.
- `[0-9]`, not `\d`, because `\d` matches Unicode digits (`CUST-١٢٣٤٥` would pass).
- No normalization. Lowercase, surrounding whitespace, and trailing newlines are rejected rather than silently fixed.

**`amount`**
- `> 0` and `≤ 10,000.00`. The cap is a sanity bound I added.
- At most 2 decimal places, so `12.345` is rejected.
- `NaN` and `±Infinity` are rejected. The SDK's JSON parser accepts those non-standard literals, so the schema has to.
- Strict mode rejects `"10.00"` and booleans. Remember that in Python `True` is an `int`.
- JSON integers like `10` are accepted, since JSON has no int/float distinction.

**`reason`**
- 10 to 500 characters.
- The 10-character minimum applies after stripping, so `"   short    "` and all-whitespace strings fail.
- Control characters other than tab and newline are rejected. Unicode is fine.

Every failing field is reported in a single response, not just the first.

## Mock data

There are three in-memory customers:

- `CUST-00001`: balance 500.00
- `CUST-12345`: balance 120.50
- `CUST-99999`: balance 0.00

Refunds reduce the refundable balance. State resets on restart. Swapping in a real store means replacing `get_customer_record` and `trigger_refund`. The protocol layer doesn't change.

## Caveat

`_answer_unparseable` uses `create_context_streams` from `mcp.shared._context_streams`, which is a private module. If an SDK upgrade breaks it, the parse-error tests will catch it.
