"""
End-to-end tests for v0.3's Hook + LayeredMemory + Agent integration.

These exercise the actual ``Agent.run()`` path with a controllable model
and verify the interaction between:

- ``LayeredMemoryStore`` (per-session memory)
- Lifecycle hook dispatch (RequestStart, BeforeClassify, AfterClassify, BeforeRespond, RequestEnd)
- ``TopicChangeHook`` triggering an L4 summary refresh
- ``add_turn()`` running at end of run() instead of paired add_message
"""

from __future__ import annotations

from typing import Any

import pytest

from swiftagentx import (
    Agent,
    DummyModelClient,
    SwiftAgentConfig,
)
from swiftagentx.core.hooks import (
    HookContext,
    HookEvent,
    HookResult,
    PythonHook,
)
from swiftagentx.core.memory_layers import LayeredMemory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeRouterModel(DummyModelClient):
    """A DummyModelClient subclass that returns a configurable classifier verdict."""

    def __init__(self, *, classify_response: str = "level=3",
                 chat_response: str = "answer", api_key: str = "k", model: str = "fake"):
        super().__init__(api_key=api_key, model=model)
        self.classify_response = classify_response
        self.chat_response = chat_response
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]], **kw: Any):  # type: ignore[override]
        self.calls.append(messages)
        joined = " ".join(m.get("content", "") for m in messages)
        if "Classify the user" in joined or "Respond with ONLY: continuing" in joined:
            # Either classifier or TopicChange prompt.
            if "Respond with ONLY: continuing" in joined:
                # TopicChange prompt — default to continuing.
                from swiftagentx.core.model_client import ModelResponse
                return ModelResponse(content="continuing", model=self.model)
            from swiftagentx.core.model_client import ModelResponse
            return ModelResponse(content=self.classify_response, model=self.model)
        from swiftagentx.core.model_client import ModelResponse
        return ModelResponse(content=self.chat_response, model=self.model)


# ---------------------------------------------------------------------------
# Agent uses LayeredMemoryStore (replaces SessionMemory)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_memory_is_layered_store() -> None:
    from swiftagentx.core.memory_layers import LayeredMemoryStore
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(enable_cache=False))
    assert isinstance(agent.memory, LayeredMemoryStore)


@pytest.mark.asyncio
async def test_run_adds_turn_to_per_session_memory() -> None:
    agent = Agent(model=FakeRouterModel(chat_response="Hi back."),
                  config=SwiftAgentConfig(enable_cache=False,
                                          memory_enable_topic_change_hook=False))
    await agent.run("hello", user_id="u1", session_id="s1")
    mem: LayeredMemory = await agent.memory.get("s1", "u1")
    assert mem.total_turns_added == 1
    assert mem.l2[-1].user_input == "hello"
    assert mem.l2[-1].assistant_response == "Hi back."


@pytest.mark.asyncio
async def test_run_separates_memory_per_session() -> None:
    agent = Agent(model=FakeRouterModel(chat_response="A"),
                  config=SwiftAgentConfig(enable_cache=False,
                                          memory_enable_topic_change_hook=False))
    await agent.run("hello", user_id="u1", session_id="sA")
    await agent.run("hello", user_id="u1", session_id="sB")
    mem_a = await agent.memory.get("sA", "u1")
    mem_b = await agent.memory.get("sB", "u1")
    assert mem_a.total_turns_added == 1
    assert mem_b.total_turns_added == 1
    # Two sessions: same user but different session_id → not shared state.
    assert id(mem_a) != id(mem_b)


# ---------------------------------------------------------------------------
# Hook dispatch during run()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_hooks_fire_in_order() -> None:
    agent = Agent(model=FakeRouterModel(chat_response="answer"),
                  config=SwiftAgentConfig(enable_cache=False,
                                          memory_enable_topic_change_hook=False))

    seen: list[str] = []

    async def make_handler(label: str):
        async def handler(ctx: HookContext) -> HookResult:
            seen.append(label)
            return HookResult()
        return handler

    for event, label in [
        (HookEvent.SESSION_START, "session_start"),
        (HookEvent.REQUEST_START, "request_start"),
        (HookEvent.BEFORE_CLASSIFY, "before_classify"),
        (HookEvent.AFTER_CLASSIFY, "after_classify"),
        (HookEvent.BEFORE_RESPOND, "before_respond"),
        (HookEvent.REQUEST_END, "request_end"),
    ]:
        agent.hooks.register(PythonHook(label, {event}, await make_handler(label)))

    await agent.run("hello", user_id="u1", session_id="s1")

    assert seen == [
        "session_start", "request_start",
        "before_classify", "after_classify",
        "before_respond", "request_end",
    ]


@pytest.mark.asyncio
async def test_session_start_only_fires_on_first_turn() -> None:
    agent = Agent(model=FakeRouterModel(chat_response="answer"),
                  config=SwiftAgentConfig(enable_cache=False,
                                          memory_enable_topic_change_hook=False))

    fires: list[int] = []

    async def handler(ctx: HookContext) -> HookResult:
        fires.append(ctx.memory.total_turns_added if ctx.memory else -1)
        return HookResult()

    agent.hooks.register(PythonHook("ss", {HookEvent.SESSION_START}, handler))
    await agent.run("first", user_id="u1", session_id="s1")
    await agent.run("second", user_id="u1", session_id="s1")
    assert len(fires) == 1


@pytest.mark.asyncio
async def test_before_classify_hook_can_short_circuit() -> None:
    agent = Agent(model=FakeRouterModel(chat_response="LLM-answer"),
                  config=SwiftAgentConfig(enable_cache=False,
                                          memory_enable_topic_change_hook=False))

    async def short(ctx: HookContext) -> HookResult:
        return HookResult(action="short_circuit", answer="hook-answer")

    agent.hooks.register(
        PythonHook("short", {HookEvent.BEFORE_CLASSIFY}, short)
    )
    response = await agent.run("hello", user_id="u1", session_id="s1")
    assert response.answer == "hook-answer"
    assert response.metadata.get("hook_short_circuit") is True

    # The short-circuited turn still records add_turn so memory stays consistent.
    mem = await agent.memory.get("s1", "u1")
    assert mem.total_turns_added == 1


# ---------------------------------------------------------------------------
# TopicChangeHook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_topic_change_hook_skips_when_memory_empty() -> None:
    """First turn ever — no prior context, hook should not call LLM."""
    fake = FakeRouterModel(chat_response="A")
    agent = Agent(model=fake,
                  config=SwiftAgentConfig(enable_cache=False,
                                          memory_enable_topic_change_hook=True))
    await agent.run("first ever", user_id="u1", session_id="s1")
    # The TopicChangeHook should NOT have triggered a separate classify call,
    # because memory was empty on entry. We can't directly count, but we can
    # verify summarize wasn't called by checking the memory state.
    mem = await agent.memory.get("s1", "u1")
    assert mem.l4_summary == ""
    assert mem.last_summarized_at is None


@pytest.mark.asyncio
async def test_topic_change_hook_can_be_disabled() -> None:
    agent = Agent(model=FakeRouterModel(),
                  config=SwiftAgentConfig(enable_cache=False,
                                          memory_enable_topic_change_hook=False))
    assert agent.hooks.list_hooks(HookEvent.BEFORE_CLASSIFY) == []


@pytest.mark.asyncio
async def test_topic_change_hook_registered_by_default() -> None:
    agent = Agent(model=FakeRouterModel(),
                  config=SwiftAgentConfig())
    assert "TopicChangeHook" in agent.hooks.list_hooks(HookEvent.BEFORE_CLASSIFY)


# ---------------------------------------------------------------------------
# Subclass override path (backwards compat with v0.2 lifecycle methods)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subclass_override_lifecycle_methods_still_work() -> None:
    """v0.2 users subclassed Agent and overrode on_request_start etc. —
    that pattern must keep working alongside the new Hook system."""

    calls: list[str] = []

    class MyAgent(Agent):
        async def on_request_start(self, context):
            calls.append("on_request_start")

        async def on_before_classify(self, context):
            calls.append("on_before_classify")

        async def on_before_respond(self, context, answer):
            calls.append("on_before_respond")
            return answer

    agent = MyAgent(model=FakeRouterModel(chat_response="hi"),
                    config=SwiftAgentConfig(enable_cache=False,
                                            memory_enable_topic_change_hook=False))
    await agent.run("hello", user_id="u1", session_id="s1")
    assert calls == [
        "on_request_start", "on_before_classify", "on_before_respond",
    ]
