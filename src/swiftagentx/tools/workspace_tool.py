"""
Built-in tool that reads back files written to the caller's session
workspace — the read-back side of context offload (see
``core/context_offload.py``). Registered automatically on every ``Agent``
so a ReAct thought or a Scenario tool-chain step can pull back a large
tool result that was offloaded out of context.

Reads are chunked (``max_chars``, with an ``offset``) and the tool is
``offload_exempt``: an offloaded result is by definition larger than the
offload threshold, so re-offloading what this tool returns would mean the
model can never actually get its content back. Bounding the chunk here is
what makes the exemption safe.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import AgentContext, Tool, ToolOutput

# Duck-typed: anything with an async `open(session_id) -> Workspace`. Kept
# untyped against `core.workspace.WorkspaceBackend` on purpose — tools/
# does not depend on core/ elsewhere in this codebase.
BackendProvider = Callable[[], Any]

DEFAULT_READ_CHUNK_CHARS = 4000


class WorkspaceReadTool(Tool):
    """Reads a chunk of a file previously written to the session workspace."""

    def __init__(
        self,
        backend_provider: BackendProvider,
        max_chars: int = DEFAULT_READ_CHUNK_CHARS,
    ):
        super().__init__(
            name="workspace_read",
            description=(
                "Read back a file from your session workspace, e.g. a large "
                f"tool result that was offloaded out of context. Returns at most "
                f"{max_chars} characters per call; pass 'offset' to continue "
                "reading. Input: {'path': 'tool_outputs/....txt', 'offset': 0}."
            ),
            category="workspace",
        )
        self._backend_provider = backend_provider
        self._max_chars = max_chars
        self.offload_exempt = True

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
                    "offset": {
                        "type": "integer",
                        "description": "Character offset to start reading from (default 0).",
                    },
                },
                "required": ["path"],
            },
        }

    async def execute(self, context: AgentContext, **kwargs: Any) -> ToolOutput:
        path = kwargs.get("path", "")
        try:
            offset = max(0, int(kwargs.get("offset", 0) or 0))
        except (TypeError, ValueError):
            return ToolOutput(success=False, result=None, error="offset must be an integer")

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

        text = data.decode("utf-8", errors="replace")
        chunk = text[offset:offset + self._max_chars]
        remaining = len(text) - (offset + len(chunk))
        if remaining > 0:
            chunk += (
                f"\n[{remaining} more characters. Continue with "
                f'workspace_read(path="{path}", offset={offset + len(chunk)}).]'
            )
        return ToolOutput(success=True, result=chunk)
