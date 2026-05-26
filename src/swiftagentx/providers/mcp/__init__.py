"""Model Context Protocol (MCP) integration for SwiftAgentX.

Exposes:

- :class:`MCPClient` — JSON-RPC client over stdio or SSE.
- :class:`MCPTool` — adapter that makes an MCP-server tool behave like a
  native :class:`swiftagentx.Tool`.
- :class:`MCPServerSpec` — declarative config for a single server.

Use ``agent.register_mcp_server(...)`` (see :mod:`swiftagentx.core.agent`)
to spin up a server, discover its tools, and add them to the agent's
:class:`ToolRegistry`. From that point forward, Scenarios and the ReAct
loop can use MCP tools indistinguishably from native ones.
"""

from .client import MCPClient, MCPClientError, MCPServerSpec, MCPTransport
from .tool import MCPTool

__all__ = [
    "MCPClient",
    "MCPClientError",
    "MCPServerSpec",
    "MCPTransport",
    "MCPTool",
]
