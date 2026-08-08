"""
D4 — Chain execution state persistence (checkpoint / resume).

A long Scenario chain interrupted mid-run — a process crash, or a step that
fails and is retried after a fix — must be able to resume from the last
completed step-group instead of re-running the whole chain from scratch.
`ScenarioCheckpoint` persists `{group_index, collected, failed_steps}`
through the same `Workspace` abstraction D3 already uses for context
offload, so this reuses an existing storage seam rather than introducing a
new one.

"Simulating a process exit" here means: state must survive the *engine*
object (and, in the Agent-level test, the *Agent* object) being thrown away
and a fresh one built — only the workspace backend (which is what actually
persists to disk in production via `LocalDiskWorkspaceBackend`) carries
state across that boundary.
"""

from __future__ import annotations

from typing import Any

import pytest

from swiftagentx import (
    Agent,
    DummyModelClient,
    InMemoryWorkspaceBackend,
    ScenarioCheckpoint,
    ScenarioConfig,
    ScenarioEngine,
    SessionContext,
    ToolChainStep,
    ToolExecutor,
    ToolOutput,
    ToolOutputType,
    ToolRegistry,
)


def _direct_scenario_env() -> tuple[Any, Any, Any]:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    context = SessionContext(session_id="s1", user_id="u1", user_input="go")
    return registry, executor, context


class _RecordingTool:
    """Records every invocation; always succeeds."""

    def __init__(self, name: str, result: str = "ok") -> None:
        self.name = name
        self.description = "records calls"
        self.category = "test"
        self.output_type = ToolOutputType.LLM_PROCESSED
        self.timeout_seconds = 5
        self.max_retries = 1
        self.calls = 0
        self._result = result

    def get_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def validate_input(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        self.calls += 1
        return ToolOutput(success=True, result=self._result)


class _FlakyTool:
    """Fails until `fail_times` calls have happened, then succeeds."""

    def __init__(self, name: str, fail_times: int, result: str = "recovered") -> None:
        self.name = name
        self.description = "fails then recovers"
        self.category = "test"
        self.output_type = ToolOutputType.LLM_PROCESSED
        self.timeout_seconds = 5
        self.max_retries = 1
        self.calls = 0
        self._fail_times = fail_times
        self._result = result

    def get_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def validate_input(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        self.calls += 1
        if self.calls <= self._fail_times:
            return ToolOutput(success=False, result=None, error="boom")
        return ToolOutput(success=True, result=self._result)


# ---------------------------------------------------------------------------
# Unit — ScenarioCheckpoint save/load/clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_load_returns_none_when_never_saved() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    checkpoint = ScenarioCheckpoint(ws, "my_scenario")
    assert await checkpoint.load() is None


@pytest.mark.asyncio
async def test_checkpoint_save_then_load_roundtrips() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    checkpoint = ScenarioCheckpoint(ws, "my_scenario")

    await checkpoint.save(group_index=2, collected={"city": "beijing"}, failed_steps=["flaky"])
    state = await checkpoint.load()

    assert state == {"group_index": 2, "collected": {"city": "beijing"}, "failed_steps": ["flaky"]}


@pytest.mark.asyncio
async def test_checkpoint_clear_removes_saved_state() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    checkpoint = ScenarioCheckpoint(ws, "my_scenario")
    await checkpoint.save(group_index=1, collected={}, failed_steps=[])

    await checkpoint.clear()

    assert await checkpoint.load() is None


# ---------------------------------------------------------------------------
# Engine — resume skips completed steps, retries only the failed one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_checkpoint_arg_behaves_exactly_as_before() -> None:
    """Backward compatibility: omitting `checkpoint` must not change
    behaviour or require a workspace at all."""
    registry, executor, context = _direct_scenario_env()
    step1 = _RecordingTool("step1")
    registry.register(step1)  # type: ignore[arg-type]

    engine = ScenarioEngine()
    scenario = ScenarioConfig(
        name="plain", triggers=[], tool_chain=[ToolChainStep(tool="step1")],
    )
    result = await engine.execute_config(scenario, "plain", context, executor)

    assert result.success
    assert step1.calls == 1


@pytest.mark.asyncio
async def test_interrupted_chain_resumes_without_rerunning_completed_steps() -> None:
    """A 3-step chain where step 2 fails: the first run stops after step 1
    succeeded and step 2 failed. A *fresh* engine instance (simulating a
    process restart) bound to the same checkpoint then resumes — step 1
    must not run again, and once step 2 is fixed the chain completes."""
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    registry, executor, context = _direct_scenario_env()

    step1 = _RecordingTool("step1", result="one")
    step2 = _FlakyTool("step2", fail_times=1, result="two")
    step3 = _RecordingTool("step3", result="three")
    registry.register(step1)  # type: ignore[arg-type]
    registry.register(step2)  # type: ignore[arg-type]
    registry.register(step3)  # type: ignore[arg-type]

    scenario = ScenarioConfig(
        name="chain", triggers=[],
        tool_chain=[
            ToolChainStep(tool="step1", extract_to="r1"),
            ToolChainStep(tool="step2", extract_to="r2"),
            ToolChainStep(tool="step3", extract_to="r3"),
        ],
        output_type="direct",
    )

    # Run 1 — process A. step2 fails, chain stops (fail_fast default).
    engine_a = ScenarioEngine()
    checkpoint_a = ScenarioCheckpoint(ws, "chain")
    result_1 = await engine_a.execute_config(
        scenario, "chain", context, executor, checkpoint=checkpoint_a,
    )
    assert not result_1.success
    assert step1.calls == 1
    assert step2.calls == 1
    assert step3.calls == 0

    state = await checkpoint_a.load()
    assert state is not None
    assert state["group_index"] == 1  # retry step2 (index 1), not step1
    assert state["collected"]["r1"] == "one"

    # Run 2 — a brand new engine + checkpoint object, bound to the same
    # workspace: this is the "resume after process exit" case.
    engine_b = ScenarioEngine()
    checkpoint_b = ScenarioCheckpoint(ws, "chain")
    result_2 = await engine_b.execute_config(
        scenario, "chain", context, executor, checkpoint=checkpoint_b,
    )

    assert result_2.success
    assert result_2.result == "three"
    assert step1.calls == 1, "step1 must not re-run on resume"
    assert step2.calls == 2, "step2 retried once (failed, then succeeded)"
    assert step3.calls == 1

    # A finished chain leaves no checkpoint behind.
    assert await checkpoint_b.load() is None


@pytest.mark.asyncio
async def test_successful_chain_clears_checkpoint() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    registry, executor, context = _direct_scenario_env()
    registry.register(_RecordingTool("step1"))  # type: ignore[arg-type]

    engine = ScenarioEngine()
    checkpoint = ScenarioCheckpoint(ws, "solo")
    scenario = ScenarioConfig(name="solo", triggers=[], tool_chain=[ToolChainStep(tool="step1")])

    result = await engine.execute_config(scenario, "solo", context, executor, checkpoint=checkpoint)

    assert result.success
    assert await checkpoint.load() is None


@pytest.mark.asyncio
async def test_best_effort_failure_still_advances_checkpoint_past_group() -> None:
    """With `on_group_failure="best_effort"`, a failed group does not block
    the chain, so resume must not retry it — the checkpoint should already
    be past it once the chain finishes."""
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    registry, executor, context = _direct_scenario_env()
    flaky = _FlakyTool("flaky", fail_times=99)  # always fails within this test
    step2 = _RecordingTool("step2")
    registry.register(flaky)  # type: ignore[arg-type]
    registry.register(step2)  # type: ignore[arg-type]

    engine = ScenarioEngine()
    checkpoint = ScenarioCheckpoint(ws, "best_effort_chain")
    scenario = ScenarioConfig(
        name="be", triggers=[],
        tool_chain=[ToolChainStep(tool="flaky"), ToolChainStep(tool="step2")],
        on_group_failure="best_effort",
    )

    result = await engine.execute_config(scenario, "be", context, executor, checkpoint=checkpoint)

    assert result.success
    assert flaky.calls == 1
    assert step2.calls == 1
    # Chain ran to completion (best-effort) — checkpoint cleared, not left
    # pointing back at the permanently-failing step.
    assert await checkpoint.load() is None


# ---------------------------------------------------------------------------
# Agent integration — checkpointing is automatic for registered Scenarios
# ---------------------------------------------------------------------------


class _ScenarioForceAgent(Agent):
    def __init__(self, scenario_id: str, **kw: Any) -> None:
        super().__init__(**kw)
        self._forced_scenario = scenario_id

    async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
        from swiftagentx.core.router import IntentLevel, IntentResult
        return IntentResult(level=IntentLevel.SCENARIO,
                            scenario=self._forced_scenario, confidence=1.0)


@pytest.mark.asyncio
async def test_agent_resumes_interrupted_scenario_across_fresh_agent_instance() -> None:
    """End-to-end: two separate `Agent` instances sharing one workspace
    backend (the persistent substrate in production) act like "before" and
    "after" a process restart. The second instance must not repeat the
    step the first instance already completed."""
    shared_backend = InMemoryWorkspaceBackend()

    step1 = _RecordingTool("step1", result="one")
    step2 = _FlakyTool("step2", fail_times=1, result="two")

    def _make_agent() -> _ScenarioForceAgent:
        agent = _ScenarioForceAgent(
            "chain",
            model=DummyModelClient(api_key="k", model="d"),
        )
        agent.workspace_backend = shared_backend
        agent.tool_registry.register(step1)  # type: ignore[arg-type]
        agent.tool_registry.register(step2)  # type: ignore[arg-type]
        agent.register_scenario("chain", ScenarioConfig(
            name="chain", description="", triggers=["chain"],
            tool_chain=[
                ToolChainStep(tool="step1", extract_to="r1"),
                ToolChainStep(tool="step2", extract_to="r2"),
            ],
            output_type="direct",
        ))
        return agent

    agent_a = _make_agent()
    response_1 = await agent_a.run("go", session_id="shared-session")
    assert "couldn't complete" in response_1.answer
    assert step1.calls == 1
    assert step2.calls == 1

    agent_b = _make_agent()
    response_2 = await agent_b.run("go", session_id="shared-session")

    assert response_2.answer == "two"
    assert step1.calls == 1, "resumed run must not repeat the completed step"
    assert step2.calls == 2
