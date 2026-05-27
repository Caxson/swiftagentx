"""
Built-in hooks that interact with :mod:`swiftagentx.core.memory_layers`.

Currently ships:

- :class:`TopicChangeHook` — a semantic hook that runs before intent
  classification, asks the LIGHT model whether the current user input
  starts a new topic, and (if yes) calls ``memory.summarize()`` to fold
  L3 into L4 before the rest of the request handling sees the memory.

The architecture document at ``docs/architecture-v0.3.md`` describes
why this hook exists.
"""

from __future__ import annotations

import logging
from typing import Any

from .hooks import HookContext, HookEvent, HookResult, SemanticHook
from .memory_layers import LayeredMemory

logger = logging.getLogger(__name__)


class TopicChangeHook(SemanticHook):
    """
    Detects topic changes and refreshes the rolling memory summary.

    The hook is wired to the ``BEFORE_CLASSIFY`` lifecycle event so the
    summary is up-to-date by the time the intent router (and the
    scenario/ReAct paths that follow) sees the layered memory.

    Implementation:

    1. ``evaluate`` calls the LIGHT model with a small prompt asking
       *"is this input continuing the prior thread or starting a new topic?"*
       The decision is made on the L2 verbatim window alone — that's the
       only window short enough to keep the LIGHT-model classification
       cheap.
    2. When the answer is ``new_topic``, ``handle`` invokes
       ``memory.summarize()`` so the agent's next prompt rendering will
       include the freshly-rolled L4 summary, with L3 cleared.

    The hook is intentionally **forgiving**: any LLM error, missing model,
    or unparseable response is logged at WARNING and treated as "no topic
    change" — we never want a flaky hook to break the main request path.
    """

    events = {HookEvent.BEFORE_CLASSIFY}

    DEFAULT_PROMPT = (
        "You are checking whether a user is changing topic in a multi-turn "
        "conversation with an AI assistant.\n\n"
        "Most recent dialog (oldest first):\n"
        "<recent>\n{recent}\n</recent>\n\n"
        "Current user input:\n"
        "<current>\n{current}\n</current>\n\n"
        "Is the current input continuing the same conversation thread, or "
        "starting a NEW topic / unrelated request?\n\n"
        "Respond with EXACTLY one word: 'continuing' or 'new_topic'."
    )

    def __init__(
        self,
        *,
        name: str = "TopicChangeHook",
        prompt_template: str | None = None,
        recent_turns: int = 4,
        model_attr: str = "light_model",
    ) -> None:
        super().__init__(name=name, events=self.events)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.recent_turns = recent_turns
        self.model_attr = model_attr

    # ------------------------------------------------------------------
    # SemanticHook contract
    # ------------------------------------------------------------------

    async def evaluate(self, context: HookContext) -> bool:
        memory = context.memory
        if not isinstance(memory, LayeredMemory):
            return False

        # No prior turns means there can't be a topic change — there's no
        # prior topic. Skip the LLM call.
        if not memory.l2 and not memory.l3 and not memory.l4_summary:
            return False

        model = getattr(context.agent, self.model_attr, None)
        if model is None:
            logger.warning(
                "TopicChangeHook: agent has no %s — cannot evaluate.",
                self.model_attr,
            )
            return False

        recent_block = self._format_recent(memory)
        prompt = self.prompt_template.format(
            recent=recent_block or "(no prior turns)",
            current=context.user_input or "",
        )

        try:
            response = await model.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=12,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TopicChangeHook LLM call failed (%s) — assuming continuing.",
                type(exc).__name__,
            )
            return False

        verdict = response.content.strip().lower()
        return "new_topic" in verdict

    async def handle(self, context: HookContext) -> HookResult:
        memory = context.memory
        if not isinstance(memory, LayeredMemory):
            return HookResult()
        # Promote ALL of L2 → L3 before summarising. Otherwise on a topic
        # boundary with a half-full L2, summarize() would find L3 empty
        # and silently no-op — the old topic would keep bleeding into the
        # new one via recent-turn replay (dogfood Friction #B-5).
        moved = await memory.flush_l2_to_l3()
        logger.info(
            "TopicChangeHook: topic change detected for session %s — "
            "promoted %d L2 turn(s) to L3, folding into L4.",
            context.session_id, moved,
        )
        new_summary = await memory.summarize()
        return HookResult(
            action="continue",
            metadata={
                "topic_change": True,
                "l2_turns_flushed": moved,
                "summary_refreshed": new_summary is not None,
            },
        )

    # ------------------------------------------------------------------

    def _format_recent(self, memory: LayeredMemory) -> str:
        last_n = memory.l2[-self.recent_turns:] if memory.l2 else []
        return "\n\n".join(turn.to_prompt_block() for turn in last_n)
