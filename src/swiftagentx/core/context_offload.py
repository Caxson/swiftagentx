"""
Context offload for large tool results (docs/OPTIMIZATION_PLAN.md, D3).

A tool result inlined verbatim into the ReAct / Scenario context can blow
past a useful prompt budget on a single call (a big JSON dump, a long file
read, ...). Above ``threshold`` characters, :func:`offload_if_large` writes
the full text to the caller's session workspace and returns a short preview
plus a workspace file reference instead. The ``workspace_read`` tool
(``tools/workspace_tool.py``) is the read-back path — it is a normal
registered tool, so both the ReAct loop and a Scenario tool chain step can
call it to pull the full content back on demand, in bounded chunks.
"""

from __future__ import annotations

import uuid
from typing import Any

from .workspace import Workspace


def stringify(value: Any) -> str:
    """Render a tool result the same way ReAct/Scenario already do for
    context — strings pass through, everything else gets ``str()``."""
    return value if isinstance(value, str) else str(value)


def offload_key(prefix: str) -> str:
    """Build a collision-free workspace key for one offloaded result.

    Callers name the *kind* of result (``react_search_2``); the random
    suffix is what keeps a second turn, a retry, or a concurrent request in
    the same session from silently overwriting an earlier file whose
    reference the model may still be holding.
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def truncate_inline(text: str, limit: int) -> str:
    """Degrade gracefully when the workspace is unavailable: keep the head
    of the text inline and say how much was dropped. Bounded context still
    beats failing a request whose tool has already run."""
    return (
        f"{text[:limit]}\n[Truncated: {len(text) - limit} more characters "
        "dropped — workspace unavailable, full result not stored.]"
    )


async def offload_if_large(
    value: Any,
    *,
    workspace: Workspace,
    key: str,
    threshold: int,
    preview_chars: int = 500,
) -> str:
    """Return the text to inline into LLM-facing context for ``value``.

    ``threshold <= 0`` disables offloading — the stringified value is
    always returned unchanged. Otherwise, text at or under ``threshold``
    characters passes through unchanged; longer text is written to
    ``tool_outputs/{key}.txt`` in ``workspace`` and a preview + reference
    is returned instead.
    """
    text = stringify(value)
    if threshold <= 0 or len(text) <= threshold:
        return text

    relative = f"tool_outputs/{key}.txt"
    await workspace.write(relative, text)
    preview = text[:preview_chars]
    return (
        f"[Tool output offloaded: {len(text)} chars written to workspace "
        f'file \'{relative}\'. Call workspace_read(path="{relative}") to '
        f"read the full content if needed.\nPreview ({preview_chars} chars):\n"
        f"{preview}]"
    )
