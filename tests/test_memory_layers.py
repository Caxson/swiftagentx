"""
Tests for the v0.3 4-layer LayeredMemory.

Covers:

- L2 verbatim window + overflow into L3
- L3 cap forces summarization
- Cadence trigger (every N turns)
- Manual ``summarize()`` call (the path a TopicChangeHook will use)
- ``render_for_prompt`` layer ordering and section formatting
- Backend persistence round-trip (InMemoryBackend)
- Idempotent ``load()`` + ``summarize()`` with no L3 turns
- Summarizer absent → graceful no-op + warning, L3 preserved
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from swiftagentx.core.memory_layers import (
    DialogTurn,
    InMemoryBackend,
    LayeredMemory,
    LayeredMemoryConfig,
)

# ---------------------------------------------------------------------------
# Helpers — deterministic fake summarizer
# ---------------------------------------------------------------------------


def make_summarizer() -> tuple[Any, list[tuple[str, list[DialogTurn]]]]:
    """Return (summarizer, calls). calls records every invocation."""

    calls: list[tuple[str, list[DialogTurn]]] = []

    async def summarizer(prev: str, new_turns: list[DialogTurn]) -> str:
        calls.append((prev, list(new_turns)))
        bullet_list = "; ".join(f"u={t.user_input}/a={t.assistant_response}" for t in new_turns)
        return f"PREV[{prev or 'empty'}] + NEW[{bullet_list}]"

    return summarizer, calls


# ---------------------------------------------------------------------------
# L2 window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_keeps_most_recent_turns_verbatim() -> None:
    summarizer, _ = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=4, summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    for i in range(3):
        await mem.add_turn(f"q{i}", f"a{i}")
    assert len(mem.l2) == 3
    assert mem.l3 == []
    assert mem.l2[0].user_input == "q0"
    assert mem.l2[-1].user_input == "q2"


@pytest.mark.asyncio
async def test_l2_overflow_pushes_oldest_into_l3() -> None:
    summarizer, _ = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=4, l3_max_size=10,
                                   summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    for i in range(7):
        await mem.add_turn(f"q{i}", f"a{i}")

    assert len(mem.l2) == 4
    assert [t.user_input for t in mem.l2] == ["q3", "q4", "q5", "q6"]
    assert len(mem.l3) == 3
    assert [t.user_input for t in mem.l3] == ["q0", "q1", "q2"]


# ---------------------------------------------------------------------------
# L3 cap and cadence triggers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l3_overflow_forces_summarize() -> None:
    summarizer, calls = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=2, l3_max_size=3,
                                   summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    # 6 turns: 2 stay in L2, 4 push into L3 — exceeds l3_max_size=3.
    for i in range(6):
        await mem.add_turn(f"q{i}", f"a{i}")

    assert len(calls) == 1, "summarize should fire exactly once"
    assert mem.l4_summary != "", "L4 should now hold the summary"
    assert mem.l3 == [], "L3 should be empty after summarize folds it in"
    assert len(mem.l2) == 2


@pytest.mark.asyncio
async def test_cadence_trigger_fires_every_n_turns() -> None:
    summarizer, calls = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=2, l3_max_size=100,
                                   summarize_every_n_turns=3,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    # The first cadence boundary is turn 3 (3 total turns added).
    for i in range(5):
        await mem.add_turn(f"q{i}", f"a{i}")
    # Turns 3 fires summarize (with one folded L3 turn), turn 5 has nothing
    # to fold but is still triggered — we exercise the no-op path.
    assert len(calls) >= 1
    # At least one of the calls should have folded ≥1 turn.
    assert any(new_turns for _, new_turns in calls)


# ---------------------------------------------------------------------------
# Manual summarize (TopicChangeHook path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manual_summarize_folds_l3_and_clears_it() -> None:
    summarizer, calls = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=2, l3_max_size=100,
                                   summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    for i in range(5):
        await mem.add_turn(f"q{i}", f"a{i}")

    folded = await mem.summarize()
    assert folded is not None
    assert mem.l3 == []
    assert mem.l4_summary == folded
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_summarize_is_noop_when_l3_empty() -> None:
    summarizer, calls = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    result = await mem.summarize()
    assert result is None
    assert calls == []


@pytest.mark.asyncio
async def test_summarize_no_summarizer_leaves_l3_intact() -> None:
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=1, l3_max_size=100,
                                   summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=None,
    )
    for i in range(3):
        await mem.add_turn(f"q{i}", f"a{i}")

    assert len(mem.l3) == 2  # nothing summarized
    result = await mem.summarize()
    assert result is None
    assert len(mem.l3) == 2


@pytest.mark.asyncio
async def test_summarize_includes_previous_summary_on_second_run() -> None:
    summarizer, calls = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=1, l3_max_size=100,
                                   summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    for i in range(3):
        await mem.add_turn(f"q{i}", f"a{i}")
    first = await mem.summarize()
    assert first is not None

    for i in range(3, 6):
        await mem.add_turn(f"q{i}", f"a{i}")
    second = await mem.summarize()
    assert second is not None

    # Second call's prev_summary should be the first call's output.
    prev_on_second = calls[1][0]
    assert prev_on_second == first


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_render_for_prompt_emits_all_four_layers() -> None:
    summarizer, _ = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=2, l3_max_size=100,
                                   summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    for i in range(4):
        await mem.add_turn(f"q{i}", f"a{i}")
    await mem.summarize()  # populate L4

    # Add 2 fresh turns so L2 and L3 are both populated.
    for i in range(4, 6):
        await mem.add_turn(f"q{i}", f"a{i}")
    # Now L2 has q4,q5; L3 has whatever rolled out since the summary.
    # Confirm at minimum: l4 present, l2 present, current_question present.
    rendered = mem.render_for_prompt(current_input="next question?")

    assert "<personal_history>" in rendered
    assert "<recent_dialog>" in rendered
    assert "<current_question>" in rendered
    assert "next question?" in rendered

    # Ordering: personal_history precedes recent_dialog precedes current_question.
    idx_l4 = rendered.index("<personal_history>")
    idx_l2 = rendered.index("<recent_dialog>")
    idx_l1 = rendered.index("<current_question>")
    assert idx_l4 < idx_l2 < idx_l1


@pytest.mark.asyncio
async def test_render_skips_empty_layers() -> None:
    mem = LayeredMemory("s1", "u1")
    rendered = mem.render_for_prompt(current_input="hello?")
    assert "<personal_history>" not in rendered
    assert "<recent_context" not in rendered
    assert "<recent_dialog>" not in rendered
    assert "<current_question>" in rendered


# ---------------------------------------------------------------------------
# Backend persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backend_round_trip() -> None:
    summarizer, _ = make_summarizer()
    backend = InMemoryBackend()
    mem = LayeredMemory(
        "session-abc", "user-1",
        backend=backend,
        config=LayeredMemoryConfig(l2_size=2, l3_max_size=100,
                                   summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    for i in range(4):
        await mem.add_turn(f"q{i}", f"a{i}")
    await mem.summarize()

    # Reload into a fresh LayeredMemory pointing at the same backend.
    fresh = LayeredMemory("session-abc", "user-1", backend=backend)
    await fresh.load()

    assert len(fresh.l2) == 2
    assert fresh.l4_summary != ""
    assert fresh.total_turns_added == 4


@pytest.mark.asyncio
async def test_load_is_idempotent() -> None:
    mem = LayeredMemory("s1", "u1")
    await mem.load()
    await mem.load()
    await mem.load()
    assert mem.l2 == []


@pytest.mark.asyncio
async def test_clear_resets_all_layers() -> None:
    summarizer, _ = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=2, l3_max_size=100,
                                   summarize_every_n_turns=0,
                                   summarize_in_background=False),
        summarizer=summarizer,
    )
    for i in range(4):
        await mem.add_turn(f"q{i}", f"a{i}")
    await mem.summarize()
    await mem.clear()

    assert mem.l2 == []
    assert mem.l3 == []
    assert mem.l4_summary == ""
    assert mem.total_turns_added == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_reflects_layer_sizes() -> None:
    mem = LayeredMemory("s1", "u1")
    for i in range(3):
        await mem.add_turn(f"q{i}", f"a{i}")
    s = mem.stats()
    assert s["session_id"] == "s1"
    assert s["user_id"] == "u1"
    assert s["l2_count"] == 3
    assert s["l3_count"] == 0
    assert s["total_turns_added"] == 3


# ---------------------------------------------------------------------------
# Background summarization smoke test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_summarize_eventually_runs() -> None:
    summarizer, calls = make_summarizer()
    mem = LayeredMemory(
        "s1", "u1",
        config=LayeredMemoryConfig(l2_size=1, l3_max_size=2,
                                   summarize_every_n_turns=0,
                                   summarize_in_background=True),
        summarizer=summarizer,
    )
    # 4 turns: 1 in L2, 3 spill to L3 — overflows l3_max_size=2, triggers
    # a background summarize task.
    for i in range(4):
        await mem.add_turn(f"q{i}", f"a{i}")

    # Yield to the event loop until the background task finishes.
    for _ in range(20):
        if calls:
            break
        await asyncio.sleep(0.01)

    assert len(calls) >= 1
