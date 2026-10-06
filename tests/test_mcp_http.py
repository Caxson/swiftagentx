"""Streamable HTTP through the real MCP client and Agent tool registry."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from swiftagentx import Agent, DummyModelClient
from swiftagentx.providers.mcp import MCPClient, MCPClientError, MCPServerSpec


@pytest.mark.parametrize("sse", [False, True])
async def test_http_registry_execution_and_headers(monkeypatch: pytest.MonkeyPatch, sse: bool) -> None:
    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        message = json.loads(request.content)
        method = message["method"]
        if method == "initialize":
            assert "mcp-session-id" not in request.headers
            result = {"protocolVersion": "2025-03-26", "serverInfo": {"name": "fixture"}}
            headers = {"Mcp-Session-Id": "fixture-session"}
        else:
            assert request.headers["mcp-session-id"] == "fixture-session"
            assert request.headers["mcp-protocol-version"] == "2025-03-26"
            headers = {}
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "tools/list":
                result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
            else:
                assert method == "tools/call"
                assert message["params"]["arguments"] in (
                    {"objective": "MCP documentation", "search_queries": ["MCP specification"]},
                    {"urls": ["https://modelcontextprotocol.io"]},
                )
                result = {"content": [{"type": "text", "text": "MCP documentation: https://modelcontextprotocol.io"}]}
        payload = {"jsonrpc": "2.0", "id": message["id"], "result": result}
        if sse:
            # Ignore comments and notifications; support multi-line data events.
            encoded = json.dumps(payload, indent=2)
            event = "\n".join(f"data: {line}" for line in encoded.splitlines())
            return httpx.Response(
                200,
                headers={**headers, "Content-Type": "text/event-stream"},
                text=f': keepalive\n\ndata: {{"jsonrpc":"2.0","method":"progress"}}\n\n{event}\n\n',
            )
        return httpx.Response(200, headers=headers, json=payload)

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    agent = Agent(model=DummyModelClient(api_key="unused", model="dummy"))
    try:
        names = await agent.register_mcp_server("parallel", transport="streamable_http", url="https://fixture.test/mcp")
        assert names == ["parallel.web_search", "parallel.web_fetch"]
        context = SimpleNamespace(variables={}, user_input="MCP", session_id="fixture")
        search = agent.tool_registry.get(names[0])
        fetch = agent.tool_registry.get(names[1])
        assert search is not None and fetch is not None
        outputs = await asyncio.gather(
            search.execute(context, objective="MCP documentation", search_queries=["MCP specification"]),
            fetch.execute(context, urls=["https://modelcontextprotocol.io"]),
        )
        assert all(output.success and "https://modelcontextprotocol.io" in output.result for output in outputs)
        assert all(request.headers["user-agent"] == "swiftagentx/0.4.1" for request in requests)
        assert all("authorization" not in r.headers and "x-api-key" not in r.headers for r in requests)
        assert [json.loads(r.content)["method"] for r in requests[:3]] == [
            "initialize",
            "notifications/initialized",
            "tools/list",
        ]
    finally:
        await agent.shutdown_mcp_servers()


@pytest.mark.parametrize("cancel", [False, True])
async def test_http_timeout_and_cancellation_cleanup(monkeypatch: pytest.MonkeyPatch, cancel: bool) -> None:
    started = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        return httpx.Response(200)

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    client = MCPClient(
        MCPServerSpec(name="slow", transport="streamable_http", url="https://fixture.test/mcp", initialize_timeout=0.05)
    )
    task = asyncio.create_task(client.start())
    await started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else MCPClientError):
        await task
    assert not client._pending
    assert not client._started
    assert client._transport._client.is_closed


async def test_http_protocol_error_and_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        msg = json.loads(request.content)
        if msg["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": msg["id"], "result": {}})
        if "id" not in msg:
            return httpx.Response(202)
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32602, "message": "invalid arguments"}}
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(respond), **kw))
    async with MCPClient(
        MCPServerSpec(name="errors", transport="streamable_http", url="https://fixture.test/mcp")
    ) as client:
        for _ in range(2):
            with pytest.raises(MCPClientError, match="invalid arguments"):
                await client.call_tool("web_search", {})
            assert not client._pending


def test_http_requires_url_and_preserves_default() -> None:
    assert MCPServerSpec(name="default").transport == "stdio"
    with pytest.raises(MCPClientError, match="no url"):
        MCPClient(MCPServerSpec(name="missing", transport="streamable_http"))
