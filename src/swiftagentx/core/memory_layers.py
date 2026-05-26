"""
4-layer Persistent Memory for SwiftAgentX v0.3.

The architecture document at ``docs/architecture-v0.3.md`` is the source
of truth for design decisions; this module implements them.

Layer semantics
---------------

L1  Current question
    The single ``user_input`` being processed right now. Not stored here
    — passed directly to the prompt at render time.

L2  Verbatim recent dialog
    The most recent ``l2_size`` turns (default 4), stored as full
    ``DialogTurn`` records with both user input and assistant response.
    Always given to the model with the highest priority.

L3  Reference window
    Turns older than L2 that have not yet been folded into L4. Capped
    at ``l3_max_size`` entries (default 6); overflow forces an immediate
    summarize. Given to the model as "reference" context.

L4  Rolling summary
    A single text field. Updated incrementally: when summarize() runs,
    it re-prompts a LIGHT model with ``(previous_summary + folded_turns)``
    and atomically swaps the result in.

Summarization triggers
----------------------

Either of:

- **Cadence**: every ``summarize_every_n_turns`` turns
  (default 5; counted by ``total_turns_added``).
- **Manual / hook**: external code (e.g. ``TopicChangeHook``) calls
  ``await memory.summarize()`` directly when a semantic condition fires.

Both paths run the same ``summarize()`` coroutine.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class DialogTurn(BaseModel):
    """A single user/assistant exchange, kept verbatim in L2 and L3."""

    user_input: str
    assistant_response: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_prompt_block(self) -> str:
        return (
            f"User: {self.user_input}\n"
            f"Assistant: {self.assistant_response}"
        )


class MemorySnapshot(BaseModel):
    """Serialisable form of a session's full layered memory."""

    session_id: str
    user_id: str
    l2_turns: list[DialogTurn] = Field(default_factory=list)
    l3_turns: list[DialogTurn] = Field(default_factory=list)
    l4_summary: str = ""
    total_turns_added: int = 0
    last_summarized_at: datetime | None = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayeredMemoryConfig:
    """Tunable knobs for ``LayeredMemory``."""

    l2_size: int = 4
    l3_max_size: int = 6
    summarize_every_n_turns: int = 5
    summarize_in_background: bool = True
    summarize_prompt_template: str = (
        "You are maintaining a conversation memory for an AI agent.\n\n"
        "Previous summary of older dialog (may be empty for the first run):\n"
        "<previous_summary>\n{previous_summary}\n</previous_summary>\n\n"
        "New dialog turns to fold into the summary (oldest first):\n"
        "<new_turns>\n{new_turns}\n</new_turns>\n\n"
        "Produce an updated single-paragraph summary that:\n"
        "1. Preserves every concrete fact, preference, decision, and "
        "commitment from the previous summary.\n"
        "2. Incorporates the salient facts from the new turns.\n"
        "3. Stays under 400 tokens.\n"
        "4. Is written in third person ('the user asked …').\n\n"
        "Respond with ONLY the new summary text — no preamble, no markup."
    )


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class MemoryBackend:
    """
    Persistence interface for ``LayeredMemory``.

    The default implementation keeps everything in-process. Subclass and
    plug in Redis, Postgres, etc. for production deployments.
    """

    async def load(self, session_id: str) -> MemorySnapshot | None:
        return None

    async def save(self, snapshot: MemorySnapshot) -> None:
        return None


class InMemoryBackend(MemoryBackend):
    """Process-local dict-backed memory store. Default for tests and dev."""

    def __init__(self) -> None:
        self._store: dict[str, MemorySnapshot] = {}

    async def load(self, session_id: str) -> MemorySnapshot | None:
        return self._store.get(session_id)

    async def save(self, snapshot: MemorySnapshot) -> None:
        self._store[snapshot.session_id] = snapshot


# ---------------------------------------------------------------------------
# Summarizer protocol
# ---------------------------------------------------------------------------


# A summarizer is anything that maps (previous_summary, new_turns) -> str.
# In practice this is an LLM call against the LIGHT model, but we factor it
# out so tests can inject a deterministic fake.
Summarizer = Callable[[str, list[DialogTurn]], Awaitable[str]]


def _default_summarizer_factory(
    light_model: Any,
    template: str,
) -> Summarizer:
    """Build a summarizer that calls the agent's LIGHT model."""

    async def _summarize(prev: str, new_turns: list[DialogTurn]) -> str:
        joined = "\n\n".join(t.to_prompt_block() for t in new_turns)
        prompt = template.format(previous_summary=prev or "(empty)",
                                 new_turns=joined or "(none)")
        response = await light_model.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return response.content.strip()

    return _summarize


# ---------------------------------------------------------------------------
# LayeredMemory
# ---------------------------------------------------------------------------


class LayeredMemory:
    """
    The 4-layer memory store for one session.

    Threading model: every public method that mutates state takes the
    instance-level asyncio lock. Concurrent ``add_turn`` and
    ``summarize`` calls are safe; reads are not gated (they may observe
    a turn that's mid-fold, which is acceptable).
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        *,
        backend: MemoryBackend | None = None,
        config: LayeredMemoryConfig | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.backend = backend or InMemoryBackend()
        self.config = config or LayeredMemoryConfig()
        self._summarizer = summarizer

        self.l2: list[DialogTurn] = []
        self.l3: list[DialogTurn] = []
        self.l4_summary: str = ""
        self.total_turns_added: int = 0
        self.last_summarized_at: datetime | None = None
        self._lock = asyncio.Lock()
        self._loaded = False

    # ---- lifecycle ------------------------------------------------------

    async def load(self) -> None:
        """Hydrate from the backend (idempotent)."""
        if self._loaded:
            return
        snapshot = await self.backend.load(self.session_id)
        if snapshot is not None:
            self.l2 = list(snapshot.l2_turns)
            self.l3 = list(snapshot.l3_turns)
            self.l4_summary = snapshot.l4_summary
            self.total_turns_added = snapshot.total_turns_added
            self.last_summarized_at = snapshot.last_summarized_at
        self._loaded = True

    async def save(self) -> None:
        await self.backend.save(self._snapshot())

    def _snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            session_id=self.session_id,
            user_id=self.user_id,
            l2_turns=list(self.l2),
            l3_turns=list(self.l3),
            l4_summary=self.l4_summary,
            total_turns_added=self.total_turns_added,
            last_summarized_at=self.last_summarized_at,
        )

    def set_summarizer(self, summarizer: Summarizer) -> None:
        """Inject the summarizer (typically built from agent.light_model)."""
        self._summarizer = summarizer

    # ---- add_turn -------------------------------------------------------

    async def add_turn(
        self,
        user_input: str,
        assistant_response: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a completed user/assistant exchange.

        The turn lands in L2. If L2 overflows, the oldest L2 turn rolls
        into L3. If L3 overflows OR the cadence threshold is hit,
        summarization is triggered (foreground or background per config).
        """
        async with self._lock:
            await self.load()

            turn = DialogTurn(
                user_input=user_input,
                assistant_response=assistant_response,
                metadata=metadata or {},
            )
            self.l2.append(turn)
            self.total_turns_added += 1

            # L2 overflow → roll oldest L2 into L3.
            while len(self.l2) > self.config.l2_size:
                self.l3.append(self.l2.pop(0))

            needs_summarize = (
                len(self.l3) > self.config.l3_max_size
                or self._cadence_fires()
            )

        if needs_summarize:
            if self.config.summarize_in_background:
                # Fire-and-forget; failures are logged but don't break the
                # caller. We still persist the post-add_turn state below.
                asyncio.create_task(self._safe_summarize())
            else:
                await self.summarize()

        # Persist the post-add_turn snapshot (without waiting for the
        # background summarize to finish).
        await self.save()

    def _cadence_fires(self) -> bool:
        every = self.config.summarize_every_n_turns
        if every <= 0:
            return False
        return self.total_turns_added > 0 and self.total_turns_added % every == 0

    # ---- summarize ------------------------------------------------------

    async def summarize(self) -> str | None:
        """
        Fold L3 into L4. Returns the new summary, or None if nothing to do.

        Idempotent: calling twice in a row with no new turns is a no-op
        the second time.
        """
        async with self._lock:
            await self.load()

            if not self.l3:
                return None
            if self._summarizer is None:
                logger.warning(
                    "LayeredMemory.summarize() called without a summarizer "
                    "configured for session %s — keeping L3 as-is.",
                    self.session_id,
                )
                return None

            to_fold = list(self.l3)
            prev = self.l4_summary

        # The actual LLM call is OUTSIDE the lock — we don't want to
        # block other writes while waiting on the network.
        try:
            new_summary = await self._summarizer(prev, to_fold)
        except Exception as exc:
            logger.warning(
                "Summarizer raised %s for session %s; keeping previous summary.",
                type(exc).__name__, self.session_id,
            )
            return None

        async with self._lock:
            # Drop only the turns we folded — other turns may have rolled
            # in while we were summarizing.
            still_present = [t for t in to_fold if t in self.l3]
            for t in still_present:
                self.l3.remove(t)
            self.l4_summary = new_summary
            self.last_summarized_at = datetime.now()

        await self.save()
        return new_summary

    async def _safe_summarize(self) -> None:
        try:
            await self.summarize()
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Background summarize failed for session %s: %s",
                self.session_id, exc,
            )

    # ---- rendering ------------------------------------------------------

    def render_for_prompt(
        self,
        *,
        current_input: str | None = None,
        include_l4: bool = True,
        include_l3: bool = True,
        include_l2: bool = True,
    ) -> str:
        """
        Produce the layered-memory section of a prompt.

        Sections are emitted in order least-prominent → most-prominent so
        the model treats L4 as background context and L1 as the focal
        question, matching the cache-friendly prompt-ordering design.
        """
        sections: list[str] = []

        if include_l4 and self.l4_summary:
            sections.append(
                "<personal_history>\n"
                f"{self.l4_summary}\n"
                "</personal_history>"
            )

        if include_l3 and self.l3:
            l3_block = "\n\n".join(t.to_prompt_block() for t in self.l3)
            sections.append(
                "<recent_context for_reference>\n"
                f"{l3_block}\n"
                "</recent_context>"
            )

        if include_l2 and self.l2:
            l2_block = "\n\n".join(t.to_prompt_block() for t in self.l2)
            sections.append(
                "<recent_dialog>\n"
                f"{l2_block}\n"
                "</recent_dialog>"
            )

        if current_input is not None:
            sections.append(
                "<current_question>\n"
                f"{current_input}\n"
                "</current_question>"
            )

        return "\n\n".join(sections)

    # ---- introspection / stats -----------------------------------------

    def stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "l2_count": len(self.l2),
            "l3_count": len(self.l3),
            "l4_chars": len(self.l4_summary),
            "total_turns_added": self.total_turns_added,
            "last_summarized_at": (
                self.last_summarized_at.isoformat() if self.last_summarized_at else None
            ),
        }

    async def clear(self) -> None:
        async with self._lock:
            self.l2.clear()
            self.l3.clear()
            self.l4_summary = ""
            self.total_turns_added = 0
            self.last_summarized_at = None
        await self.save()
