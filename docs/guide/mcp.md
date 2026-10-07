# MCP tools

`Agent.register_mcp_server()` discovers an MCP server's tools and adds them to
`agent.tool_registry` as `{server_name}.{tool_name}`. The cookbook below retrieves
these registered tools with `agent.tool_registry.get(name)` and calls
`await tool.execute(context, **arguments)` directly.

The current ReAct agent loop cannot invoke these dotted tool names: its action
parser reads only word characters, so `Action: parallel.web_search` is parsed as
`parallel` rather than the registered `parallel.web_search`. Registration alone
does not make these MCP tools callable through the agent loop. Use the direct
registry calls shown in the cookbook.

## Parallel search and fetch

Install the optional HTTP dependency from this checkout:

```bash
pip install -e '.[openai]'
```

Register the anonymous [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
endpoint with the opt-in Streamable HTTP transport:

```python
from swiftagentx import Agent, DummyModelClient

agent = Agent(model=DummyModelClient(api_key="unused", model="dummy"))
try:
    names = await agent.register_mcp_server(
        "parallel",
        transport="streamable_http",
        url="https://search.parallel.ai/mcp",
        initialize_timeout=10.0,
        call_timeout=30.0,
    )
    # parallel.web_search and parallel.web_fetch are now in the registry.
finally:
    await agent.shutdown_mcp_servers()
```

Run the complete example from a checkout:

```bash
pip install -e '.[openai]'
python examples/cookbook/parallel_search.py
```

The example executes both registered tools and prints search excerpts and fetched
page content. It uses a dummy model and makes no inference calls. The anonymous
endpoint needs no Parallel API key, but has rate limits. Using a real model for
routing or answers is a separate configuration and may incur inference charges.
Search arguments include `objective` and `search_queries`; fetch takes `urls`
and an optional `objective`. The example reuses one `session_id` for both calls.
Consult the server's discovered schemas for current arguments and limits.

## Transport scope

`stdio` remains the default, and the existing `sse` transport remains available.
Select `streamable_http` explicitly for a single HTTP MCP endpoint. It uses
JSON-RPC POSTs, accepts JSON and SSE responses, retains the server's session ID,
sends the negotiated protocol version and initialized notification, and uses
`User-Agent: swiftagentx/0.4.1` on discovery and tool calls. The existing timeouts
bound the complete HTTP exchange, including response streaming.

This is a request/response client for tools. Background GET streams,
server-initiated requests, stream resumption, and authentication configuration
are not supported. Closing the client releases its local HTTP connections;
server sessions expire according to the server's policy. Always shut down MCP
servers in `finally` to release connections, including after tool failures.
