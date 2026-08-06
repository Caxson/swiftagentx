"""
D3 — Tool result context offload.

Tool outputs above `SwiftAgentConfig.context_offload_threshold` characters
must not be inlined verbatim into the ReAct / Scenario context sent to the
LLM. Instead they're written to the session workspace and the context
keeps a short preview + a workspace file reference; the `workspace_read`
tool is the read-back path for both the ReAct loop and a Scenario chain.

benchmark 待本地补测（云端环境无法跑真实 LLM/网络延迟对比，token 用量对比需要
真实模型调用）。
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
    WorkspaceReadTool,
    offload_if_large,
)

# ---------------------------------------------------------------------------
# Unit — offload_if_large()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_small_value_passes_through_unchanged() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    text = await offload_if_large("short", workspace=ws, key="k1", threshold=100)
    assert text == "short"
    assert await ws.list() == []


@pytest.mark.asyncio
async def test_large_value_is_written_to_workspace_and_referenced() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    full = "x" * 500
    text = await offload_if_large(
        full, workspace=ws, key="mytool_1", threshold=100, preview_chars=20,
    )
    assert full not in text
    assert "tool_outputs/mytool_1.txt" in text
    assert "500 chars" in text
    assert "x" * 20 in text  # preview prefix present

    stored = await ws.read("tool_outputs/mytool_1.txt")
    assert stored is not None
    assert stored.decode("utf-8") == full


@pytest.mark.asyncio
async def test_non_string_value_is_stringified_before_measuring() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    payload = {"data": list(range(200))}
    text = await offload_if_large(payload, workspace=ws, key="k2", threshold=50)
    assert "tool_outputs/k2.txt" in text
    stored = await ws.read("tool_outputs/k2.txt")
    assert stored.decode("utf-8") == str(payload)


@pytest.mark.asyncio
async def test_zero_threshold_disables_offload() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    full = "y" * 10_000
    text = await offload_if_large(full, workspace=ws, key="k3", threshold=0)
    assert text == full
    assert await ws.list() == []


# ---------------------------------------------------------------------------
# Unit — WorkspaceReadTool (the read-back path)
# ---------------------------------------------------------------------------


class _FakeContext:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.user_input = ""
        self.variables: dict[str, Any] = {}


@pytest.mark.asyncio
async def test_workspace_read_tool_reads_back_offloaded_file() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    await offload_if_large("z" * 300, workspace=ws, key="big", threshold=10)

    tool = WorkspaceReadTool(lambda: backend)
    result = await tool.execute(_FakeContext("s1"), path="tool_outputs/big.txt")
    assert result.success
    assert result.result == "z" * 300


@pytest.mark.asyncio
async def test_workspace_read_tool_errors_on_missing_file() -> None:
    backend = InMemoryWorkspaceBackend()
    tool = WorkspaceReadTool(lambda: backend)
    result = await tool.execute(_FakeContext("s1"), path="tool_outputs/nope.txt")
    assert not result.success
    assert "nope.txt" in (result.error or "")


def test_workspace_read_tool_validate_input_requires_path() -> None:
    tool = WorkspaceReadTool(lambda: InMemoryWorkspaceBackend())
    assert not tool.validate_input()
    assert tool.validate_input(path="tool_outputs/a.txt")


# ---------------------------------------------------------------------------
# Agent wiring — workspace_read is auto-registered
# ---------------------------------------------------------------------------


def test_agent_auto_registers_workspace_read_tool() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"))
    assert "workspace_read" in agent.tool_registry.list_tools()


# ---------------------------------------------------------------------------
# Agent integration — ReAct loop offloads a large observation
# ---------------------------------------------------------------------------


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
    """Records every prompt handed to `chat()` so tests can inspect exactly
    what context reached the (simulated) LLM."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.prompts: list[str] = []

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        self.prompts.append(messages[-1]["content"])
        return ModelResponse(content="[recorded]", model=self.model, tokens_used=1)


class _OneShotReactAgent(Agent):
    """Forces REACT, calls `big_tool` once, then lets the loop run out of
    `max_iterations` (2) so it falls through to `_generate_final_answer` —
    the real (non-overridden) code path that builds the LLM-facing prompt
    from `accumulated_context`. Deterministic, no real LLM needed."""

    async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
        from swiftagentx.core.router import IntentLevel, IntentResult
        return IntentResult(level=IntentLevel.REACT, confidence=1.0)

    async def _generate_thought(self, context: Any, model: Any, accumulated: str) -> str:  # type: ignore[override]
        if not accumulated:
            return 'Thought: fetch it.\nAction: big_tool\nAction Input: {}'
        return "Thought: nothing more to do."


@pytest.mark.asyncio
async def test_react_loop_offloads_large_observation_out_of_context() -> None:
    payload = "A" * 5000
    recorder = _RecordingModelClient(api_key="k", model="d")
    agent = _OneShotReactAgent(
        model=recorder,
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            max_iterations=2,
            context_offload_threshold=200, context_offload_preview_chars=30,
        ),
    )
    agent.workspace_backend = InMemoryWorkspaceBackend()
    agent.tool_registry.register(_BigResultTool(payload))  # type: ignore[arg-type]

    response = await agent.run("go", session_id="react-s1")

    # The final-answer prompt (last recorded chat call) must not contain
    # the raw 5000-char payload, only a preview + workspace reference.
    final_prompt = recorder.prompts[-1]
    assert payload not in final_prompt
    assert "tool_outputs/react_big_tool_1_" in final_prompt

    # But the full content is still recoverable from the workspace.
    ws = await agent.workspace_backend.open("react-s1")
    stored_files = await ws.list()
    assert len(stored_files) == 1
    stored = await ws.read(stored_files[0])
    assert stored.decode("utf-8") == payload
    assert response.answer == "[recorded]"


@pytest.mark.asyncio
async def test_react_loop_keeps_small_observation_inline() -> None:
    payload = "small result"
    recorder = _RecordingModelClient(api_key="k", model="d")
    agent = _OneShotReactAgent(
        model=recorder,
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            max_iterations=2,
            context_offload_threshold=200,
        ),
    )
    agent.workspace_backend = InMemoryWorkspaceBackend()
    agent.tool_registry.register(_BigResultTool(payload))  # type: ignore[arg-type]

    await agent.run("go", session_id="react-s2")

    final_prompt = recorder.prompts[-1]
    assert payload in final_prompt
    assert "offloaded" not in final_prompt


# ---------------------------------------------------------------------------
# Agent integration — Scenario formatting offloads a large step result
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
async def test_scenario_llm_processed_offloads_large_result() -> None:
    payload = "B" * 5000
    recorder = _RecordingModelClient(api_key="k", model="d")
    agent = _ScenarioForceAgent(
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
        output_type="llm_processed",
    ))

    await agent.run("dump it", session_id="scn-s1")

    final_prompt = recorder.prompts[-1]
    assert payload not in final_prompt
    assert "tool_outputs/scenario_dump_" in final_prompt

    ws = await agent.workspace_backend.open("scn-s1")
    stored_files = await ws.list()
    assert len(stored_files) == 1
    stored = await ws.read(stored_files[0])
    assert stored.decode("utf-8") == payload


@pytest.mark.asyncio
async def test_scenario_direct_output_is_never_offloaded() -> None:
    """`output_type="direct"` returns the raw tool result straight to the
    caller — it never goes through the LLM, so offloading it would corrupt
    the user-visible answer."""
    payload = "C" * 5000
    agent = _ScenarioForceAgent(
        "dump",
        model=DummyModelClient(api_key="k", model="d"),
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

    response = await agent.run("dump it", session_id="scn-s2")
    assert response.answer == payload


# ---------------------------------------------------------------------------
# Read-back must terminate: `workspace_read` is exempt from offload and
# returns a bounded chunk, so the model can actually get its content back
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_read_returns_bounded_chunk_with_next_offset() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    await ws.write("tool_outputs/big.txt", "D" * 5000)
    tool = WorkspaceReadTool(lambda: backend, max_chars=1000)

    out = await tool.execute(_Ctx("s1"), path="tool_outputs/big.txt")

    assert out.success
    body, _, note = out.result.partition("\n[")
    assert body == "D" * 1000
    assert "offset=1000" in note  # tells the model how to continue


@pytest.mark.asyncio
async def test_workspace_read_offset_reads_the_tail() -> None:
    backend = InMemoryWorkspaceBackend()
    ws = await backend.open("s1")
    await ws.write("tool_outputs/big.txt", "E" * 1500)
    tool = WorkspaceReadTool(lambda: backend, max_chars=1000)

    out = await tool.execute(_Ctx("s1"), path="tool_outputs/big.txt", offset=1000)

    assert out.success
    assert out.result == "E" * 500  # tail fits, no continuation note


def test_workspace_read_tool_is_offload_exempt() -> None:
    assert WorkspaceReadTool(lambda: None).offload_exempt is True


class _ReadBackReactAgent(Agent):
    """Offloads a big result, then does what the offload notice tells the
    model to do: call `workspace_read` on the referenced path."""

    async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
        from swiftagentx.core.router import IntentLevel, IntentResult
        return IntentResult(level=IntentLevel.REACT, confidence=1.0)

    async def _generate_thought(self, context: Any, model: Any, accumulated: str) -> str:  # type: ignore[override]
        if not accumulated:
            return 'Thought: fetch it.\nAction: big_tool\nAction Input: {}'
        if "workspace_read(path=" in accumulated and "Tool: workspace_read" not in accumulated:
            path = accumulated.split('workspace_read(path="')[1].split('"')[0]
            return (f'Thought: read it back.\nAction: workspace_read\n'
                    f'Action Input: {{"path": "{path}"}}')
        return "Thought: done."


@pytest.mark.asyncio
async def test_react_read_back_is_not_offloaded_again() -> None:
    """The read-back path must terminate: `workspace_read`'s own output is
    exempt from offload, so its (bounded) chunk lands in context instead of
    being written out to yet another file."""
    payload = "F" * 5000
    recorder = _RecordingModelClient(api_key="k", model="d")
    agent = _ReadBackReactAgent(
        model=recorder,
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            max_iterations=3,
            context_offload_threshold=200, context_offload_preview_chars=30,
            context_offload_read_chunk_chars=400,
        ),
    )
    agent.workspace_backend = InMemoryWorkspaceBackend()
    agent.tool_registry.register(_BigResultTool(payload))  # type: ignore[arg-type]

    await agent.run("go", session_id="react-readback")

    ws = await agent.workspace_backend.open("react-readback")
    files = await ws.list()
    assert len(files) == 1, f"read-back must not spawn another offload file: {files}"

    final_prompt = recorder.prompts[-1]
    assert "Tool: workspace_read" in final_prompt
    assert "F" * 400 in final_prompt  # the bounded chunk actually arrived


# ---------------------------------------------------------------------------
# Offload keys are unique per call — no silent cross-turn overwrite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_turns_do_not_overwrite_each_others_offload_files() -> None:
    recorder = _RecordingModelClient(api_key="k", model="d")
    agent = _OneShotReactAgent(
        model=recorder,
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            max_iterations=2,
            context_offload_threshold=200, context_offload_preview_chars=30,
        ),
    )
    agent.workspace_backend = InMemoryWorkspaceBackend()
    tool = _BigResultTool("G" * 5000)
    agent.tool_registry.register(tool)  # type: ignore[arg-type]

    await agent.run("first", session_id="react-multi")
    first_files = set(await (await agent.workspace_backend.open("react-multi")).list())

    tool._payload = "H" * 5000  # same tool, different result on turn 2
    await agent.run("second", session_id="react-multi")
    all_files = set(await (await agent.workspace_backend.open("react-multi")).list())

    assert len(all_files) == 2, "turn 2 silently overwrote turn 1's offload file"
    ws = await agent.workspace_backend.open("react-multi")
    kept = await ws.read(first_files.pop())
    assert kept.decode("utf-8") == "G" * 5000  # turn 1 content intact


# ---------------------------------------------------------------------------
# Planner fast path must offload too
# ---------------------------------------------------------------------------


class _PlanForceAgent(Agent):
    """Routes to REACT but resolves a cached plan, so `_execute_plan_steps`
    (the planner fast path) builds the LLM-facing prompt."""

    async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
        from swiftagentx.core.router import IntentLevel, IntentResult
        return IntentResult(level=IntentLevel.REACT, confidence=1.0)


@pytest.mark.asyncio
async def test_planner_fast_path_offloads_large_result() -> None:
    payload = "I" * 5000
    recorder = _RecordingModelClient(api_key="k", model="d")
    agent = _PlanForceAgent(
        model=recorder,
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            context_offload_threshold=200, context_offload_preview_chars=30,
        ),
    )
    agent.workspace_backend = InMemoryWorkspaceBackend()
    agent.tool_registry.register(_BigResultTool(payload))  # type: ignore[arg-type]

    from swiftagentx.models.schema import SessionContext
    context = SessionContext(session_id="plan-s1", user_id="u", user_input="go")
    answer = await agent._execute_plan_steps(
        [ToolChainStep(tool="big_tool")], {}, context, "big",
    )

    assert answer == "[recorded]"
    plan_prompt = recorder.prompts[-1]
    assert payload not in plan_prompt, "planner path bypassed context offload"
    assert "tool_outputs/plan_big_" in plan_prompt


# ---------------------------------------------------------------------------
# Offload failure must degrade, never fail the request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_write_failure_degrades_to_truncated_inline() -> None:
    """A full disk must not turn a successful tool call into a failed
    request — the tool already ran, side effects included."""

    class _BrokenBackend(InMemoryWorkspaceBackend):
        async def open(self, session_id: str) -> Any:
            ws = await super().open(session_id)

            async def _boom(*a: Any, **kw: Any) -> None:
                raise OSError(28, "No space left on device")

            ws.write = _boom  # type: ignore[method-assign]
            return ws

    payload = "J" * 5000
    recorder = _RecordingModelClient(api_key="k", model="d")
    agent = _OneShotReactAgent(
        model=recorder,
        config=SwiftAgentConfig(
            memory_enable_topic_change_hook=False, enable_cache=False,
            max_iterations=2,
            context_offload_threshold=200, context_offload_preview_chars=30,
        ),
    )
    agent.workspace_backend = _BrokenBackend()
    agent.tool_registry.register(_BigResultTool(payload))  # type: ignore[arg-type]

    response = await agent.run("go", session_id="react-broken")

    assert response.answer == "[recorded]"  # request survived
    final_prompt = recorder.prompts[-1]
    assert payload not in final_prompt  # still bounded
    assert "J" * 200 in final_prompt  # degraded to a truncated inline copy


class _Ctx:
    """Minimal AgentContext for direct tool execution."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.user_input = ""
        self.variables: dict[str, Any] = {}
