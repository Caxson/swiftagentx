"""
Cache-friendly prompt layout for SwiftAgentX v0.3.

Anthropic and OpenAI prompt caches reward stable *prefixes*. Build the
prompt in least-changing → most-changing order and the cache hit rate
on the prefix improves dramatically. For a high-volume deployment that
sends millions of nearly-identical prompts (the SwiftAgentX target),
this is a 30-50% cost reduction.

The canonical order:

    1. tools_section    — stable across all calls (tool catalog)
    2. system_section   — stable across all calls (role + instructions)
    3. l4_summary       — changes slowly, only when summarize() fires
    4. l3_reference     — changes per request but the tail is stable
    5. l2_recent_dialog — changes per request
    6. l1_current_input — changes per request

``PromptLayout`` is a small helper that takes those six pieces and emits
them in this order, with consistent delimiters. Nothing else uses it
yet; the next wiring commit will route :class:`PromptManager` through
this helper so all of classify / scenario / react / direct paths share
the canonical layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptLayout:
    """Assemble a chat-style prompt in cache-friendly order."""

    tools_section: str = ""
    system_section: str = ""
    l4_summary: str = ""
    l3_reference: str = ""
    l2_recent_dialog: list[dict[str, str]] | None = None
    l1_current_input: str = ""

    def as_chat_messages(self) -> list[dict[str, str]]:
        """Render in OpenAI chat-message form.

        Stable prefix lives in the first ``system`` message
        (tools + system + L4 + L3); L2 becomes alternating user/assistant
        turns; L1 is the final user message.
        """
        prefix_parts: list[str] = []
        if self.tools_section:
            prefix_parts.append(self.tools_section.rstrip())
        if self.system_section:
            prefix_parts.append(self.system_section.rstrip())
        if self.l4_summary:
            prefix_parts.append(
                f"<personal_history>\n{self.l4_summary.strip()}\n</personal_history>"
            )
        if self.l3_reference:
            prefix_parts.append(
                f"<recent_context for_reference>\n{self.l3_reference.strip()}\n"
                f"</recent_context>"
            )

        messages: list[dict[str, str]] = []
        if prefix_parts:
            messages.append({"role": "system", "content": "\n\n".join(prefix_parts)})

        for turn in (self.l2_recent_dialog or []):
            messages.append(turn)

        if self.l1_current_input:
            messages.append({"role": "user", "content": self.l1_current_input})

        return messages

    def as_single_prompt(self) -> str:
        """Flatten into a single string. For provider APIs that don't
        accept chat-message lists."""
        chunks: list[str] = []
        if self.tools_section:
            chunks.append(self.tools_section)
        if self.system_section:
            chunks.append(self.system_section)
        if self.l4_summary:
            chunks.append(
                f"<personal_history>\n{self.l4_summary}\n</personal_history>"
            )
        if self.l3_reference:
            chunks.append(
                f"<recent_context for_reference>\n{self.l3_reference}\n"
                f"</recent_context>"
            )
        if self.l2_recent_dialog:
            l2_block = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in self.l2_recent_dialog
            )
            chunks.append(f"<recent_dialog>\n{l2_block}\n</recent_dialog>")
        if self.l1_current_input:
            chunks.append(
                f"<current_question>\n{self.l1_current_input}\n</current_question>"
            )
        return "\n\n".join(chunks)

    @classmethod
    def from_agent(
        cls,
        *,
        agent: Any,
        memory: Any,
        user_input: str,
        tools_section: str = "",
        system_section: str = "",
        l2_rounds: int = 5,
    ) -> PromptLayout:
        """Convenience constructor that pulls the memory layers from a
        :class:`LayeredMemory` and tools/system from the agent."""
        l4_summary = getattr(memory, "l4_summary", "") if memory else ""
        l3_block = ""
        l2_messages: list[dict[str, str]] = []
        if memory is not None:
            l3 = getattr(memory, "l3", None) or []
            if l3:
                l3_block = "\n\n".join(t.to_prompt_block() for t in l3)
            chat = memory.to_chat_messages(
                include_l4=False, include_l3=False, l2_rounds=l2_rounds,
            )
            # to_chat_messages emits any L4+L3 system message first; we
            # explicitly disabled those — so only L2 turns come back here.
            l2_messages = chat

        return cls(
            tools_section=tools_section,
            system_section=system_section,
            l4_summary=l4_summary,
            l3_reference=l3_block,
            l2_recent_dialog=l2_messages,
            l1_current_input=user_input,
        )
