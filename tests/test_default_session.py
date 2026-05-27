"""
Regression tests for v0.3.1 frictions:

- `Agent.run()` without `session_id` shares one default session per Agent
  instance — memory persists across calls (the v0.3.0 dogfood headline bug).
- The exception handler exposes `error_class` + `error_message` in
  response metadata even when `debug=False`.
- `OpenAICompatibleProvider` raises a clear ImportError at construction
  time if httpx isn't installed (smoke-tested via monkeypatching since
  httpx IS installed in this env).
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from swiftagentx import Agent, DummyModelClient, SwiftAgentConfig


# ---------------------------------------------------------------------------
# Friction #5 — default session_id is per-Agent stable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_session_id_persists_across_calls() -> None:
    """Two calls on the same Agent without session_id share the same memory."""
    agent = Agent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False),
    )
    r1 = await agent.run("turn one")
    r2 = await agent.run("turn two")

    # Same session_id reported in both responses.
    assert r1.session_id == r2.session_id
    assert r1.session_id == agent._default_session_id

    # Underlying memory accumulates both turns.
    mem = await agent.memory.get(r1.session_id, "anonymous")
    assert mem.total_turns_added == 2
    assert [t.user_input for t in mem.l2[-2:]] == ["turn one", "turn two"]


@pytest.mark.asyncio
async def test_two_agents_get_distinct_default_sessions() -> None:
    """Each Agent instance gets its own default session — no cross-talk."""
    a1 = Agent(model=DummyModelClient(api_key="k", model="d"),
               config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    a2 = Agent(model=DummyModelClient(api_key="k", model="d"),
               config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    assert a1._default_session_id != a2._default_session_id


@pytest.mark.asyncio
async def test_explicit_session_id_still_wins() -> None:
    """Passing session_id explicitly overrides the default."""
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                          enable_cache=False))
    r1 = await agent.run("a", session_id="explicit-1")
    r2 = await agent.run("b", session_id="explicit-2")
    assert r1.session_id == "explicit-1"
    assert r2.session_id == "explicit-2"
    assert r1.session_id != agent._default_session_id


# ---------------------------------------------------------------------------
# Friction #4 — error metadata surfaces class + message even with debug=False
# ---------------------------------------------------------------------------


class _BoomTool:
    """Injected via a custom subclass override to force an exception."""

    async def execute(self, *a: Any, **kw: Any) -> Any:  # pragma: no cover
        raise RuntimeError("kaboom from tool")


class _BoomingAgent(Agent):
    async def _direct_response(self, context: Any) -> str:  # type: ignore[override]
        raise RuntimeError("kaboom in direct response")


@pytest.mark.asyncio
async def test_error_metadata_includes_class_in_non_debug_mode() -> None:
    agent = _BoomingAgent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False, debug=False),
    )
    response = await agent.run("trigger")
    # User-facing answer is still the sanitized message.
    assert "internal error" in response.answer.lower()
    # But metadata now surfaces what blew up.
    assert response.metadata.get("error_class") == "RuntimeError"
    assert "kaboom" in (response.metadata.get("error_message") or "")
    # No traceback when debug=False.
    assert "traceback" not in response.metadata


@pytest.mark.asyncio
async def test_error_metadata_includes_traceback_in_debug_mode() -> None:
    agent = _BoomingAgent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False, debug=True),
    )
    response = await agent.run("trigger")
    assert response.metadata.get("error_class") == "RuntimeError"
    assert "Traceback" in (response.metadata.get("traceback") or "")


# ---------------------------------------------------------------------------
# Friction #3 — OpenAICompatibleProvider construction fails clearly without httpx
# ---------------------------------------------------------------------------


def test_openai_provider_clear_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate `httpx` missing — provider should raise an actionable ImportError."""
    # Pretend httpx isn't installed.
    monkeypatch.setitem(sys.modules, "httpx", None)

    from swiftagentx.providers.openai_compatible import OpenAICompatibleProvider

    with pytest.raises(ImportError) as exc_info:
        OpenAICompatibleProvider(api_key="k", model="m")
    msg = str(exc_info.value)
    assert "swiftagentx[openai]" in msg or "httpx" in msg
