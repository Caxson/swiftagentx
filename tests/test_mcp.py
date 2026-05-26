"""
Tests for the MCP (Model Context Protocol) integration.

Strategy: rather than depend on a real MCP server in CI, we drive the
client with a fake transport that scripts ``tools/list`` and ``tools/call``
responses. The end-to-end path through ``MCPTool`` and
``Agent.register_mcp_server`` is verified against that fake.

A separate live subprocess test (skipped on Windows) exercises the stdio
transport with a small echo-server script written inline.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Iterable
from typing import Any

import pytest

from swiftagentx import Agent, DummyModelClient, SwiftAgentConfig
from swiftagentx.providers.mcp.client import (
    MCPClient,
    MCPClientError,
    MCPServerSpec,
)
from swiftagentx.providers.mcp.tool import MCPTool


# ---------------------------------------------------------------------------
# Fake transport — drives MCPClient deterministically
# ---------------------------------------------------------------------------


class FakeTransport:
    """Request-driven fake: each send() releases exactly one scripted response.

    The MCPClient dispatcher only registers a pending future after send()
    returns, so we delay the response until the next recv() call to avoid
    racing the future registration.
    """

    def __init__(self, scripted_responses: Iterable[dict[str, Any]]):
        self.scripted = list(scripted_responses)
        self.sent: list[dict[str, Any]] = []
        self._inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def start(self) -> None:
        return None

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        # Pair this request with the next scripted response and let the
        # dispatcher pick it up. We give the dispatcher one event-loop tick
        # to register the pending future for ``message['id']``.
        if not self.scripted:
            return
        await asyncio.sleep(0)
        response = self.scripted.pop(0)
        if "id" in message and "id" not in response:
            response = {**response, "id": message["id"]}
        self._inbox.put_nowait(response)

    async def recv(self, timeout: float) -> dict[str, Any]:
        return await asyncio.wait_for(self._inbox.get(), timeout=timeout)

    async def close(self) -> None:
        pass


def _make_client_with_fake(responses: list[dict[str, Any]]) -> MCPClient:
    spec = MCPServerSpec(name="fake", transport="stdio", command=["/bin/sh"])
    client = MCPClient.__new__(MCPClient)
    client.spec = spec
    client._transport = FakeTransport(responses)
    client._next_id = 1
    client._pending = {}
    client._dispatcher = None
    client._started = False
    client._initialized = False
    return client


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_initialize_handshake() -> None:
    client = _make_client_with_fake([
        {"jsonrpc": "2.0", "id": 1, "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "fake-mcp", "version": "0.0.1"},
        }},
    ])
    await client.start()
    assert client._initialized
    assert client._transport.sent[0]["method"] == "initialize"  # type: ignore[attr-defined]
    await client.close()


# ---------------------------------------------------------------------------
# tools/list and tools/call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tools_and_call_tool() -> None:
    client = _make_client_with_fake([
        # initialize response
        {"jsonrpc": "2.0", "id": 1, "result": {
            "serverInfo": {"name": "fake-mcp"},
        }},
        # tools/list response
        {"jsonrpc": "2.0", "id": 2, "result": {
            "tools": [{
                "name": "query",
                "description": "Run a SQL query",
                "inputSchema": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            }],
        }},
        # tools/call response
        {"jsonrpc": "2.0", "id": 3, "result": {
            "content": [{"type": "text", "text": "ok"}],
            "isError": False,
        }},
    ])

    await client.start()
    tools = await client.list_tools()
    assert tools[0]["name"] == "query"

    envelope = await client.call_tool("query", {"sql": "select 1"})
    assert envelope["isError"] is False
    assert envelope["content"][0]["text"] == "ok"

    await client.close()


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_error_raises_mcp_client_error() -> None:
    client = _make_client_with_fake([
        {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}},
        {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "no such method"}},
    ])
    await client.start()
    with pytest.raises(MCPClientError) as exc_info:
        await client.list_tools()
    assert "no such method" in str(exc_info.value)
    await client.close()


# ---------------------------------------------------------------------------
# MCPTool: envelope -> ToolOutput
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_wraps_client_call() -> None:
    client = _make_client_with_fake([
        {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}},
        # tool result for tool_a:
        {"jsonrpc": "2.0", "id": 2, "result": {
            "content": [{"type": "text", "text": "hello"}],
            "isError": False,
        }},
    ])
    await client.start()

    tool = MCPTool(client=client, descriptor={
        "name": "tool_a",
        "description": "demo tool",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    })
    assert tool.name == "fake.tool_a"

    output = await tool.execute(context=None, x=1, y="z")  # type: ignore[arg-type]
    assert output.success
    assert output.result == "hello"
    assert "mcp_envelope" in output.metadata

    await client.close()


@pytest.mark.asyncio
async def test_mcp_tool_surfaces_server_error() -> None:
    client = _make_client_with_fake([
        {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}},
        # tools/call response indicating an error
        {"jsonrpc": "2.0", "id": 2, "result": {
            "content": [{"type": "text", "text": "boom"}],
            "isError": True,
        }},
    ])
    await client.start()
    tool = MCPTool(client=client, descriptor={"name": "die", "description": ""})
    output = await tool.execute(context=None)  # type: ignore[arg-type]
    assert not output.success
    assert "boom" in (output.error or "")
    await client.close()


# ---------------------------------------------------------------------------
# Agent integration (monkey-patch register_mcp_server's client factory)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_register_mcp_server_uses_fake_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: agent.register_mcp_server() registers MCPTools in the registry."""
    from swiftagentx.providers.mcp import client as client_mod

    def _fake_build_transport(spec: MCPServerSpec) -> FakeTransport:
        return FakeTransport([
            {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "f"}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": [
                {"name": "a", "description": "tool a"},
                {"name": "b", "description": "tool b"},
            ]}},
        ])

    monkeypatch.setattr(client_mod.MCPClient, "_build_transport",
                        staticmethod(_fake_build_transport))

    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))

    registered = await agent.register_mcp_server(
        "demo", transport="stdio", command=["/bin/echo"],
    )
    assert registered == ["demo.a", "demo.b"]
    assert agent.tool_registry.get("demo.a") is not None
    assert agent.tool_registry.get("demo.b") is not None

    await agent.shutdown_mcp_servers()


# ---------------------------------------------------------------------------
# Live stdio transport — only on POSIX
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell required")
async def test_live_stdio_transport_smoke(tmp_path: Any) -> None:
    """Spawn a tiny inline 'MCP server' written as a Python script and
    verify the stdio transport actually pumps newline-delimited JSON."""
    server_script = tmp_path / "fake_server.py"
    server_script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    if msg['method'] == 'initialize':\n"
        "        print(json.dumps({'jsonrpc':'2.0','id':msg['id'],"
        "'result':{'serverInfo':{'name':'inline'}}}), flush=True)\n"
        "    elif msg['method'] == 'tools/list':\n"
        "        print(json.dumps({'jsonrpc':'2.0','id':msg['id'],"
        "'result':{'tools':[{'name':'ping','description':'pong'}]}}), flush=True)\n"
        "    elif msg['method'] == 'tools/call':\n"
        "        print(json.dumps({'jsonrpc':'2.0','id':msg['id'],"
        "'result':{'content':[{'type':'text','text':'pong'}],'isError':False}}),"
        " flush=True)\n"
    )

    spec = MCPServerSpec(
        name="inline",
        transport="stdio",
        command=[sys.executable, str(server_script)],
        initialize_timeout=5.0,
        call_timeout=5.0,
    )

    async with MCPClient(spec) as client:
        tools = await client.list_tools()
        assert tools[0]["name"] == "ping"

        envelope = await client.call_tool("ping", {})
        assert envelope["content"][0]["text"] == "pong"


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------


def test_stdio_spec_requires_command() -> None:
    spec = MCPServerSpec(name="bad", transport="stdio", command=[])
    with pytest.raises(MCPClientError):
        MCPClient._build_transport(spec)


def test_unsupported_transport_raises() -> None:
    spec = MCPServerSpec(name="bad", transport="websocket")  # type: ignore[arg-type]
    with pytest.raises(MCPClientError):
        MCPClient._build_transport(spec)
