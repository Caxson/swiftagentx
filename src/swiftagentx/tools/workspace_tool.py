"""
Built-in tool that reads back files written to the caller's session
workspace — the read-back side of context offload (see
``core/context_offload.py``). Registered automatically on every ``Agent``
so a ReAct thought or a Scenario tool-chain step can pull back a large
tool result that was offloaded out of context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import AgentContext, Tool, ToolOutput

# Duck-typed: anything with an async `open(session_id) -> Workspace`. Kept
# untyped against `core.workspace.WorkspaceBackend` on purpose — tools/
# does not depend on core/ elsewhere in this codebase.
BackendProvider = Callable[[], Any]


class WorkspaceReadTool(Tool):
    """Reads a file previously written to the caller's session workspace."""

    def __init__(self, backend_provider: BackendProvider):
        super().__init__(
            name="workspace_read",
            description=(
                "Read back a file from your session workspace, e.g. a large "
                "tool result that was offloaded out of context. Input: "
                "{'path': 'tool_outputs/....txt'}."
            ),
            category="workspace",
        )
        self._backend_provider = backend_provider

    def validate_input(self, **kwargs: Any) -> bool:
        return bool(kwargs.get("path"))

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "output_type": self.output_type.value,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path, e.g. tool_outputs/foo.txt",
                    },
                },
                "required": ["path"],
            },
        }

    async def execute(self, context: AgentContext, **kwargs: Any) -> ToolOutput:
        path = kwargs.get("path", "")
        backend = self._backend_provider()
        ws = await backend.open(context.session_id)
        try:
            data = await ws.read(path)
        except ValueError as e:
            return ToolOutput(success=False, result=None, error=str(e))
        finally:
            await ws.close()

        if data is None:
            return ToolOutput(success=False, result=None, error=f"No such workspace file: {path}")
        return ToolOutput(success=True, result=data.decode("utf-8", errors="replace"))
