"""
D3b — direct 大结果的 memory 回灌卸载.

`output_type="direct"` results become the turn's answer verbatim (D3
deliberately never offloads them — the user must get the raw payload back).
But once that answer lands in L2 verbatim memory via `add_turn`, it gets
replayed whole into every later prompt (`LayeredMemory.to_chat_messages`,
used by `Agent._direct_response`), silently reintroducing the exact context
blowup D3 was built to avoid — just one hop later, across turns instead of
within one.

This offloads the answer the same way D3 offloads oversized tool results —
write it to the session workspace, keep a preview + reference in L2 — right
before `add_turn`, on every `add_turn` call site. The user-facing answer for
the turn that produced it is untouched; only the L2 copy is bounded.
"""

from __future__ import annotations

from typing import Any

import pytest

from swiftagentx import (
    Agent,
    DummyModelClient,
    InMemoryWorkspaceBackend,
    ModelResponse,
    ScenarioConfig,
    SwiftAgentConfig,
    ToolChainStep,
    ToolOutput,
    ToolOutputType,
)


class _BigResultTool:
    """Tool whose result is deliberately larger than the test threshold."""

    def __init__(self, payload: str) -> None:
        self.name = "big_tool"
        self.description = "returns a large result"
        self.category = "test"
        self.output_type = ToolOutputType.LLM_PROCESSED
        self.timeout_seconds = 5
        self.max_retries = 1
        self._payload = payload

    def get_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def validate_input(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        return ToolOutput(success=True, result=self._payload)


class _RecordingModelClient(DummyModelClient):
    """Records every message list handed to `chat()` so tests can inspect
    exactly what context reached the (simulated) LLM across turns."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.calls: list[list[dict[str, str]]] = []

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        self.calls.append(messages)
        return ModelResponse(content="[recorded]", model=self.model, tokens_used=1)


class _ScenarioThenDirectAgent(Agent):
    """Forces a SCENARIO classification on the first call (so `big_tool`'s
    direct-output result becomes the turn's answer), then DIRECT on every
    call after that (so `_direct_response` builds a real LLM prompt from
    L2 memory) — deterministic, no real classifier/model needed."""

    def __init__(self, scenario_id: str, **kw: Any) -> None:
        super().__init__(**kw)
        self._forced_scenario = scenario_id
        self._calls = 0

    async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
        from swiftagentx.core.router import IntentLevel, IntentResult
        self._calls += 1
        if self._calls == 1:
            return IntentResult(level=IntentLevel.SCENARIO,
                                scenario=self._forced_scenario, confidence=1.0)
        return IntentResult(level=IntentLevel.DIRECT, confidence=1.0)


@pytest.mark.asyncio
async def test_direct_output_still_returns_full_answer_to_user() -> None:
    """D3's guarantee must survive D3b: the turn that produces a direct-output
    answer still gets the raw payload back, unmodified."""
    payload = "E" * 5000
    agent = _ScenarioThenDirectAgent(
        "dump",
        model=_RecordingModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            context_offload_threshold=200,
        ),
    )
    agent.workspace_backend = InMemoryWorkspaceBackend()
    agent.tool_registry.register(_BigResultTool(payload))  # type: ignore[arg-type]
    agent.register_scenario("dump", ScenarioConfig(
        name="dump", description="dumps a big result",
        triggers=["dump"],
        tool_chain=[ToolChainStep(tool="big_tool")],
        output_type="direct",
    ))

    response = await agent.run("dump it", session_id="mem-s1")
    assert response.answer == payload


@pytest.mark.asyncio
async def test_large_direct_answer_is_offloaded_out_of_l2() -> None:
    payload = "E" * 5000
    agent = _ScenarioThenDirectAgent(
        "dump",
        model=_RecordingModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            context_offload_threshold=200, context_offload_preview_chars=30,
        ),
    )
    agent.workspace_backend = InMemoryWorkspaceBackend()
    agent.tool_registry.register(_BigResultTool(payload))  # type: ignore[arg-type]
    agent.register_scenario("dump", ScenarioConfig(
        name="dump", description="dumps a big result",
        triggers=["dump"],
        tool_chain=[ToolChainStep(tool="big_tool")],
        output_type="direct",
    ))

    await agent.run("dump it", session_id="mem-s2")

    mem = await agent.memory.get("mem-s2", "anonymous")
    stored = mem.l2[-1].assistant_response
    assert payload not in stored
    assert "tool_outputs/memory_turn_" in stored

    ws = await agent.workspace_backend.open("mem-s2")
    stored_files = await ws.list()
    assert len(stored_files) == 1
    full = await ws.read(stored_files[0])
    assert full.decode("utf-8") == payload


@pytest.mark.asyncio
async def test_second_turn_prompt_does_not_replay_first_turns_large_answer() -> None:
    payload = "E" * 5000
    recorder = _RecordingModelClient(api_key="k", model="d")
    agent = _ScenarioThenDirectAgent(
        "dump",
        model=recorder,
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            context_offload_threshold=200, context_offload_preview_chars=30,
        ),
    )
    agent.workspace_backend = InMemoryWorkspaceBackend()
    agent.tool_registry.register(_BigResultTool(payload))  # type: ignore[arg-type]
    agent.register_scenario("dump", ScenarioConfig(
        name="dump", description="dumps a big result",
        triggers=["dump"],
        tool_chain=[ToolChainStep(tool="big_tool")],
        output_type="direct",
    ))

    first = await agent.run("dump it", session_id="mem-s3")
    assert first.answer == payload

    await agent.run("what did you just tell me?", session_id="mem-s3")

    # Only the second turn talks to the model (the first is a pure
    # direct-output scenario, no LLM call); its prompt carries L2 memory.
    assert len(recorder.calls) == 1
    all_prompt_text = " ".join(
        m.get("content", "") for m in recorder.calls[0]
    )
    assert payload not in all_prompt_text
    assert "tool_outputs/memory_turn_" in all_prompt_text


@pytest.mark.asyncio
async def test_small_direct_answer_stays_inline_in_memory() -> None:
    payload = "short answer"
    agent = _ScenarioThenDirectAgent(
        "dump",
        model=_RecordingModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            context_offload_threshold=200,
        ),
    )
    agent.workspace_backend = InMemoryWorkspaceBackend()
    agent.tool_registry.register(_BigResultTool(payload))  # type: ignore[arg-type]
    agent.register_scenario("dump", ScenarioConfig(
        name="dump", description="dumps a small result",
        triggers=["dump"],
        tool_chain=[ToolChainStep(tool="big_tool")],
        output_type="direct",
    ))

    await agent.run("dump it", session_id="mem-s4")

    mem = await agent.memory.get("mem-s4", "anonymous")
    assert mem.l2[-1].assistant_response == payload
    ws = await agent.workspace_backend.open("mem-s4")
    assert await ws.list() == []
