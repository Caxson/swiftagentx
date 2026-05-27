"""
Minimal MCP (Model Context Protocol) JSON-RPC client.

The MCP spec is large; this client implements just enough to bootstrap a
server, list its tools, and call them. Specifically:

- ``initialize``           — protocol handshake
- ``tools/list``           — discover available tools
- ``tools/call``           — invoke a tool by name

Two transports are supported:

- **stdio**: launch the server as a subprocess and exchange newline-
  delimited JSON-RPC over its stdin/stdout. This is the most common
  transport for local-process MCP servers.
- **SSE**: post requests to ``{url}`` and receive responses via
  Server-Sent Events on ``{url}/sse``. The implementation is a thin
  wrapper around ``httpx.AsyncClient`` and only activates when ``httpx``
  is installed (it's an optional dependency).

The client is intentionally small: a future v0.4 release may swap it for
the official ``mcp`` package once it stabilizes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors + types
# ---------------------------------------------------------------------------


class MCPClientError(RuntimeError):
    """Raised for protocol violations and unrecoverable transport failures."""


MCPTransport = Literal["stdio", "sse"]


@dataclass(frozen=True)
class MCPServerSpec:
    """Declarative configuration for one MCP server."""

    name: str
    transport: MCPTransport = "stdio"
    command: list[str] = field(default_factory=list)   # stdio
    url: str | None = None                              # sse
    env: dict[str, str] = field(default_factory=dict)
    initialize_timeout: float = 10.0
    call_timeout: float = 30.0


# ---------------------------------------------------------------------------
# Transport adapters
# ---------------------------------------------------------------------------


class _StdioTransport:
    """Newline-delimited JSON-RPC over a subprocess's stdio."""

    def __init__(self, spec: MCPServerSpec) -> None:
        if not spec.command:
            raise MCPClientError(f"stdio MCP server {spec.name} has no command")
        if shutil.which(spec.command[0]) is None and not os.path.exists(spec.command[0]):
            raise MCPClientError(
                f"MCP server {spec.name}: command {spec.command[0]!r} not found on PATH"
            )
        self.spec = spec
        self._proc: asyncio.subprocess.Process | None = None
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        env = {**os.environ, **self.spec.env}
        self._proc = await asyncio.create_subprocess_exec(
            *self.spec.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def send(self, message: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        line = (json.dumps(message) + "\n").encode("utf-8")
        async with self._write_lock:
            self._proc.stdin.write(line)
            await self._proc.stdin.drain()

    async def recv(self, timeout: float) -> dict[str, Any]:
        assert self._proc and self._proc.stdout
        async with self._read_lock:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
        if not line:
            raise MCPClientError(f"MCP server {self.spec.name} closed its stdout")
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise MCPClientError(
                f"MCP server {self.spec.name} sent non-JSON line: {line!r}"
            ) from exc

    async def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except (TimeoutError, asyncio.TimeoutError):
                self._proc.kill()
                await self._proc.wait()


class _SseTransport:
    """JSON-RPC over Server-Sent Events. Lazy-imports httpx."""

    def __init__(self, spec: MCPServerSpec) -> None:
        if not spec.url:
            raise MCPClientError(f"sse MCP server {spec.name} has no url")
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise MCPClientError(
                "SSE transport requires the [openai] / [all] extras: pip install 'swiftagentx[openai]'"
            ) from exc
        self.spec = spec
        self._client: Any = None
        self._inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        import httpx
        self._client = httpx.AsyncClient(timeout=None)
        self._reader_task = asyncio.create_task(self._reader())

    async def _reader(self) -> None:
        assert self._client is not None
        url = f"{self.spec.url.rstrip('/')}/sse"
        try:
            async with self._client.stream("GET", url) as response:
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    try:
                        await self._inbox.put(json.loads(payload))
                    except json.JSONDecodeError:
                        logger.warning(
                            "MCP SSE %s: dropped non-JSON event: %r",
                            self.spec.name, payload[:120],
                        )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP SSE %s reader stopped: %s", self.spec.name, exc)

    async def send(self, message: dict[str, Any]) -> None:
        assert self._client is not None
        url = self.spec.url
        response = await self._client.post(url, json=message)
        response.raise_for_status()

    async def recv(self, timeout: float) -> dict[str, Any]:
        return await asyncio.wait_for(self._inbox.get(), timeout=timeout)

    async def close(self) -> None:
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MCPClient:
    """
    Minimal MCP JSON-RPC client.

    Use the async context manager to ensure the transport is closed:

        async with MCPClient(spec) as client:
            tools = await client.list_tools()
            result = await client.call_tool("postgres.query", {"sql": "..."})

    Or manage start/close manually:

        client = MCPClient(spec)
        await client.start()
        try:
            ...
        finally:
            await client.close()
    """

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self._transport = self._build_transport(spec)
        self._next_id = 1
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._dispatcher: asyncio.Task[None] | None = None
        self._started = False
        self._initialized = False

    @staticmethod
    def _build_transport(spec: MCPServerSpec) -> _StdioTransport | _SseTransport:
        if spec.transport == "stdio":
            return _StdioTransport(spec)
        if spec.transport == "sse":
            return _SseTransport(spec)
        raise MCPClientError(f"Unsupported MCP transport: {spec.transport!r}")

    # ---- lifecycle ------------------------------------------------------

    async def __aenter__(self) -> MCPClient:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def start(self) -> None:
        if self._started:
            return
        await self._transport.start()
        self._started = True
        self._dispatcher = asyncio.create_task(self._dispatch_loop())
        await self._initialize()

    async def close(self) -> None:
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            try:
                await self._dispatcher
            except (asyncio.CancelledError, Exception):
                pass
        if self._started:
            await self._transport.close()
            self._started = False

    # ---- dispatch loop --------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """Read responses and resolve their matching futures."""
        try:
            while True:
                msg = await self._transport.recv(timeout=3600.0)
                rid = msg.get("id")
                if rid is None:
                    # Notifications and progress events — ignored for now.
                    continue
                fut = self._pending.pop(rid, None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP %s dispatcher stopped: %s", self.spec.name, exc)
            # Fail any in-flight callers so they don't hang forever.
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(MCPClientError(f"transport died: {exc}"))

    # ---- request/response -----------------------------------------------

    async def _request(self, method: str, params: dict[str, Any] | None = None,
                       *, timeout: float | None = None) -> dict[str, Any]:
        rid = self._next_id
        self._next_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
        }
        if params is not None:
            message["params"] = params

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[rid] = fut

        await self._transport.send(message)

        try:
            response = await asyncio.wait_for(fut, timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            self._pending.pop(rid, None)
            raise MCPClientError(f"MCP request {method!r} timed out") from exc

        if "error" in response:
            # JSON-RPC error envelope: {"code": int, "message": str, "data"?: any}.
            # Format it for an LLM observation rather than ``repr(dict)`` —
            # the LLM sees this as an observation in the ReAct loop and the
            # "code -32000" / "message" framing is easier to reason about
            # than ``{'code': -32000, 'message': '...'}`` (dogfood C-3).
            err = response["error"]
            if isinstance(err, dict):
                code = err.get("code", "?")
                message = err.get("message", "")
                detail = f" — {err['data']}" if err.get("data") else ""
                raise MCPClientError(
                    f"MCP server {self.spec.name} {method!r} failed "
                    f"(code {code}): {message}{detail}"
                )
            raise MCPClientError(
                f"MCP server {self.spec.name} {method!r} failed: {err}"
            )
        return response.get("result", {})

    # ---- handshake ------------------------------------------------------

    async def _initialize(self) -> None:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": self.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "swiftagentx", "version": "0.3.0"},
            },
            timeout=self.spec.initialize_timeout,
        )
        self._initialized = True
        logger.info(
            "MCP %s initialized: server %s, protocol %s",
            self.spec.name,
            result.get("serverInfo", {}).get("name", "unknown"),
            result.get("protocolVersion", "?"),
        )

    # ---- public API -----------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the tool descriptors the server exposes."""
        result = await self._request("tools/list", timeout=self.spec.call_timeout)
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool. Returns the raw MCP ``tools/call`` result envelope."""
        return await self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout=self.spec.call_timeout,
        )


# ---------------------------------------------------------------------------
# Convenience: spec-from-dict
# ---------------------------------------------------------------------------


def make_spec(name: str, **kwargs: Any) -> MCPServerSpec:
    """Convenience constructor that tolerates dict-style overrides."""
    return MCPServerSpec(name=name, **kwargs)
