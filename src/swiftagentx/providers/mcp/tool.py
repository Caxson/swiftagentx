"""
``MCPTool`` — adapter that makes an MCP-server tool look like a native
:class:`swiftagentx.Tool`.

A single :class:`MCPClient` instance can back many :class:`MCPTool`s.
Construction is deliberately cheap; the heavy lifting (transport bring-up,
``initialize``, ``tools/list``) happens once on the client and is shared.

The class is name-spaced via ``{server_name}.{tool_name}`` to avoid
collisions between two MCP servers that both expose a tool called
``query``.
"""

from __future__ import annotations

import logging
from typing import Any

from ...tools.base import AgentContext, Tool, ToolOutput, ToolOutputType
from .client import MCPClient, MCPClientError

logger = logging.getLogger(__name__)


class MCPTool(Tool):
    """
    Wraps a single tool exposed by an MCP server.

    The constructor takes the descriptor returned by ``tools/list`` and
    decodes it into the framework's ``Tool`` shape. ``execute`` proxies
    to :meth:`MCPClient.call_tool` and converts the result envelope into
    a :class:`ToolOutput`.
    """

    def __init__(
        self,
        client: MCPClient,
        descriptor: dict[str, Any],
        *,
        category: str = "mcp",
        output_type: ToolOutputType = ToolOutputType.LLM_PROCESSED,
    ) -> None:
        self.client = client
        self.remote_name: str = descriptor["name"]
        self.input_schema: dict[str, Any] = descriptor.get("inputSchema", {}) or {}
        qualified = f"{client.spec.name}.{self.remote_name}"
        super().__init__(
            name=qualified,
            description=descriptor.get("description", "")
            or f"MCP tool {qualified}",
            category=category,
            output_type=output_type,
            timeout_seconds=int(client.spec.call_timeout),
        )

    async def execute(self, context: AgentContext, **kwargs: Any) -> ToolOutput:
        # Scenario tool_chain passes the rendered query under the ``query`` key
        # by convention; for MCP we pass the whole kwargs dict through.
        try:
            envelope = await self.client.call_tool(self.remote_name, kwargs)
        except MCPClientError as exc:
            logger.warning(
                "MCPTool %s failed: %s", self.name, exc,
            )
            return ToolOutput(success=False, result=None, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCPTool %s raised unexpected error", self.name)
            return ToolOutput(success=False, result=None, error=str(exc))

        return self._envelope_to_output(envelope)

    @staticmethod
    def _envelope_to_output(envelope: dict[str, Any]) -> ToolOutput:
        """Translate an MCP ``tools/call`` result into a ``ToolOutput``.

        MCP returns a ``content`` array of typed parts (text, image, …).
        We collapse the text parts into a single string for the common
        case; structured callers can read the original envelope via
        ``ToolOutput.metadata['mcp_envelope']``.
        """
        is_error = bool(envelope.get("isError"))
        parts = envelope.get("content", []) or []
        text_chunks: list[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text_chunks.append(str(part.get("text", "")))
        joined = "\n".join(text_chunks)
        return ToolOutput(
            success=not is_error,
            result=joined or envelope.get("content"),
            error=joined if is_error else None,
            metadata={"mcp_envelope": envelope},
        )

    def get_schema(self) -> dict[str, Any]:
        schema = super().get_schema()
        if self.input_schema:
            schema["parameters"] = self.input_schema
        return schema
