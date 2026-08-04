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

import asyncio
import sys
import time
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
    # debug=False: only error_class for branching; NO raw error_message
    # (it can contain secrets) and NO traceback.
    assert response.metadata.get("error_class") == "RuntimeError"
    assert "error_message" not in response.metadata, (
        "debug=False must NOT leak the raw exception message to metadata — "
        "it may contain DB URLs, API keys, or other secrets."
    )
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


# ---------------------------------------------------------------------------
# Friction #6 — ReAct loop must not call the same tool with the same args twice
# ---------------------------------------------------------------------------


class _CountingTool:
    """A tool that records invocations. Built to look like a real Tool
    without importing Tool ABC (the agent looks it up by name)."""

    def __init__(self) -> None:
        from swiftagentx import ToolOutput, ToolOutputType
        self._ToolOutput = ToolOutput
        self._ToolOutputType = ToolOutputType
        self.name = "echo"
        self.description = "Echo back the input"
        self.category = "test"
        self.output_type = ToolOutputType.LLM_PROCESSED
        self.timeout_seconds = 5
        self.max_retries = 1
        self.invocations: list[str] = []

    def get_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def validate_input(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, context: Any, **kwargs: Any):
        self.invocations.append(str(sorted(kwargs.items())))
        return self._ToolOutput(success=True, result="42")


class _LoopingAgent(Agent):
    """An Agent whose intent classification always picks REACT and whose
    'thought generator' always proposes the same tool call. This lets us
    test the duplicate-action guard without real LLM cost."""

    async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
        from swiftagentx.core.router import IntentLevel, IntentResult
        return IntentResult(level=IntentLevel.REACT, confidence=1.0)

    async def _generate_thought(self, context: Any, model: Any, accumulated: str) -> str:  # type: ignore[override]
        return 'Thought: I should call echo.\nAction: echo\nAction Input: {"q": "1"}'


@pytest.mark.asyncio
async def test_react_does_not_call_same_action_twice() -> None:
    """A ReAct loop that keeps proposing the same action should execute
    that action at most once — the duplicate guard must break the loop."""
    agent = _LoopingAgent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False, max_iterations=10),
    )
    tool = _CountingTool()
    agent.tool_registry.register(tool)  # type: ignore[arg-type]

    await agent.run("anything")
    assert len(tool.invocations) == 1, (
        f"expected exactly 1 tool call (duplicate guard), got "
        f"{len(tool.invocations)}: {tool.invocations}"
    )


# ---------------------------------------------------------------------------
# Friction #7 — Scenario tool chain substitution + direct output
# ---------------------------------------------------------------------------


class _RecordingTool:
    """A Tool that records its kwargs so we can assert query_template substitution."""

    def __init__(self) -> None:
        from swiftagentx import ToolOutput, ToolOutputType
        self._ToolOutput = ToolOutput
        self._ToolOutputType = ToolOutputType
        self.name = "lookup"
        self.description = "echo input"
        self.category = "test"
        self.output_type = ToolOutputType.LLM_PROCESSED
        self.timeout_seconds = 5
        self.max_retries = 1
        self.invocations: list[dict[str, Any]] = []

    def get_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def validate_input(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, context: Any, **kwargs: Any):
        self.invocations.append(dict(kwargs))
        return self._ToolOutput(success=True, result=f"GOT: {kwargs.get('query', '<empty>')}")


class _ScenarioForceAgent(Agent):
    """Agent that classifies every request as a fixed scenario id."""

    def __init__(self, scenario_id: str, **kw: Any) -> None:
        super().__init__(**kw)
        self._forced_scenario = scenario_id

    async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
        from swiftagentx.core.router import IntentLevel, IntentResult
        return IntentResult(level=IntentLevel.SCENARIO,
                            scenario=self._forced_scenario, confidence=1.0)


@pytest.mark.asyncio
async def test_scenario_template_substitution_uses_run_kwargs() -> None:
    """`agent.run("...", city="Beijing")` should make `query_template="$city"`
    expand to "Beijing", not the literal `$city`."""
    from swiftagentx import ScenarioConfig, ToolChainStep
    agent = _ScenarioForceAgent(
        "weather",
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False),
    )
    tool = _RecordingTool()
    agent.tool_registry.register(tool)  # type: ignore[arg-type]
    agent.register_scenario("weather", ScenarioConfig(
        name="weather", description="weather lookup",
        triggers=["weather"],
        tool_chain=[ToolChainStep(tool="lookup", query_template="$city")],
        cache_ttl=60,
        output_type="direct",
    ))

    await agent.run("anything about weather", city="Beijing")
    assert tool.invocations == [{"query": "Beijing"}], (
        f"expected substituted query, got {tool.invocations}"
    )


@pytest.mark.asyncio
async def test_scenario_direct_output_returns_tool_result() -> None:
    """`output_type=\"direct\"` should return the tool's raw result, not the
    `collected` dict that holds template vars."""
    from swiftagentx import ScenarioConfig, ToolChainStep
    agent = _ScenarioForceAgent(
        "weather",
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False),
    )
    tool = _RecordingTool()
    agent.tool_registry.register(tool)  # type: ignore[arg-type]
    agent.register_scenario("weather", ScenarioConfig(
        name="weather", description="weather lookup",
        triggers=["weather"],
        tool_chain=[ToolChainStep(tool="lookup", query_template="$city")],
        cache_ttl=60,
        output_type="direct",
    ))

    response = await agent.run("anything about weather", city="Beijing")
    # The bug used to leak the {user_id, user_input, session_id, city}
    # dict as the answer. Now we get the tool's actual result.
    assert "GOT: Beijing" in response.answer
    assert "user_input" not in response.answer
    assert "session_id" not in response.answer


# ---------------------------------------------------------------------------
# Round 7 — Scenario slots extracted from natural language
# ---------------------------------------------------------------------------


def test_scenario_required_vars_extraction() -> None:
    """required_vars() = $vars in templates, minus reserved + step-produced."""
    from swiftagentx import ScenarioConfig, ToolChainStep
    sc = ScenarioConfig(
        name="x", triggers=[],
        tool_chain=[
            ToolChainStep(tool="weather", kwargs_template={"city": "$city"}),
            ToolChainStep(tool="fmt", query_template="$temp", extract_to="report"),
            ToolChainStep(tool="use", query_template="$report"),  # produced → not required
            ToolChainStep(tool="echo", query_template="$user_input"),  # reserved → excluded
        ],
    )
    assert sc.required_vars() == {"city", "temp"}


def test_router_parses_slots_from_classification() -> None:
    from swiftagentx.core.router import IntentLevel, IntentRouter
    r = IntentRouter()
    r.register_scenarios({"weather": {"name": "w", "slots": ["city"]}})
    res = r._parse_classification('level=2 scenario=weather slots={"city": "北京"}')
    assert res.level == IntentLevel.SCENARIO
    assert res.scenario == "weather"
    assert res.slots == {"city": "北京"}  # original case/CJK preserved


@pytest.mark.asyncio
async def test_scenario_slot_extracted_from_natural_language() -> None:
    """The classifier's extracted slots fill `$templates` WITHOUT the caller
    passing them as run() kwargs — so a Scenario fires from plain language
    like '北京天气怎么样' (dogfood Round 7)."""
    from swiftagentx import ScenarioConfig, ToolChainStep
    from swiftagentx.core.router import IntentLevel, IntentResult

    class _SlotAgent(Agent):
        async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
            return IntentResult(level=IntentLevel.SCENARIO, scenario="weather",
                                slots={"city": "Beijing"}, confidence=1.0)

    agent = _SlotAgent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False),
    )
    tool = _RecordingTool()
    agent.tool_registry.register(tool)  # type: ignore[arg-type]
    agent.register_scenario("weather", ScenarioConfig(
        name="weather", description="weather lookup", triggers=["weather"],
        tool_chain=[ToolChainStep(tool="lookup", query_template="$city")],
        output_type="direct",
    ))

    # No city= kwarg — the slot comes purely from classification.
    resp = await agent.run("北京天气怎么样")
    assert tool.invocations == [{"query": "Beijing"}], tool.invocations
    assert "GOT: Beijing" in resp.answer


def test_scenario_downgrades_to_react_when_slot_missing() -> None:
    """A scenario whose required slot the classifier couldn't fill must fall
    back to ReAct instead of firing a step with an unsubstituted $var."""
    from swiftagentx import ScenarioConfig, ToolChainStep
    from swiftagentx.core.router import IntentLevel, IntentResult
    from swiftagentx.models.schema import SessionContext

    agent = Agent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False),
    )
    agent.register_scenario("weather", ScenarioConfig(
        name="weather", description="weather lookup", triggers=["weather"],
        tool_chain=[ToolChainStep(tool="lookup", query_template="$city")],
        output_type="direct",
    ))
    ctx = SessionContext(session_id="s", user_id="u", user_input="weather please")

    missing = IntentResult(level=IntentLevel.SCENARIO, scenario="weather", slots={})
    assert agent._resolve_execution_level(missing, ctx) == IntentLevel.REACT

    filled = IntentResult(level=IntentLevel.SCENARIO, scenario="weather",
                          slots={"city": "Beijing"})
    assert agent._resolve_execution_level(filled, ctx) == IntentLevel.SCENARIO
    assert ctx.get_variable("city") == "Beijing"  # slot merged into context


# ---------------------------------------------------------------------------
# GitHub #1 — Scenario-level cache (cache_key_template + cache_ttl)
# ---------------------------------------------------------------------------


class _ChainCountTool:
    """Tool that counts how many times its chain actually runs."""

    def __init__(self) -> None:
        from swiftagentx import ToolOutput, ToolOutputType
        self._ToolOutput = ToolOutput
        self.name = "lookup"
        self.description = "counts runs"
        self.category = "test"
        self.output_type = ToolOutputType.LLM_PROCESSED
        self.timeout_seconds = 5
        self.max_retries = 1
        self.calls = 0

    def get_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def validate_input(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, context: Any, **kwargs: Any):
        self.calls += 1
        return self._ToolOutput(success=True, result=f"run #{self.calls}")


def _cache_scenario_agent(tool: _ChainCountTool, *, cache_key_template: str) -> Agent:
    from swiftagentx import ScenarioConfig, ToolChainStep
    agent = _ScenarioForceAgent(
        "order_status",
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(enable_cache=True,
                                memory_enable_topic_change_hook=False),
    )
    agent.tool_registry.register(tool)  # type: ignore[arg-type]
    agent.register_scenario("order_status", ScenarioConfig(
        name="Order Status", triggers=["order"],
        tool_chain=[ToolChainStep(tool="lookup", query_template="$user_id")],
        cache_key_template=cache_key_template,
        cache_ttl=1800,
        output_type="direct",
    ))
    return agent


@pytest.mark.asyncio
async def test_scenario_cache_dedups_across_phrasings() -> None:
    """With cache_key_template set, the same semantic request (different
    wording, same user) runs the tool chain once — the request-level cache
    can't dedup these because the raw input differs (GitHub #1)."""
    tool = _ChainCountTool()
    agent = _cache_scenario_agent(tool, cache_key_template="order_status_$user_id")
    for q in ["where is my order?", "order status please", "track my package"]:
        await agent.run(q, user_id="u1", session_id="s1")
    assert tool.calls == 1, f"expected 1 tool run (scenario-cached), got {tool.calls}"


@pytest.mark.asyncio
async def test_scenario_cache_is_opt_in() -> None:
    """A scenario that did NOT declare a cache_key_template is never cached,
    even though cache_ttl defaults to non-zero — caching is opt-in."""
    tool = _ChainCountTool()
    agent = _cache_scenario_agent(tool, cache_key_template="")  # no template
    for q in ["q one", "q two", "q three"]:
        await agent.run(q, user_id="u1", session_id="s1")
    assert tool.calls == 3, f"expected no caching (opt-in), got {tool.calls} runs"


@pytest.mark.asyncio
async def test_scenario_cache_isolated_per_user() -> None:
    """cache_key_template='..._$user_id' must not leak one user's cached
    scenario result to another user."""
    tool = _ChainCountTool()
    agent = _cache_scenario_agent(tool, cache_key_template="order_status_$user_id")
    await agent.run("where is my order?", user_id="alice", session_id="sa")
    await agent.run("where is my order?", user_id="bob", session_id="sb")
    assert tool.calls == 2, f"per-user keys should not share; got {tool.calls}"


# ---------------------------------------------------------------------------
# D1 — Scenario parallel step groups (structure + executor)
#
# `ScenarioConfig.tool_chain` stays a plain list; a parallel group is just
# a nested list of steps with no dependencies between them — fan out via
# `asyncio.gather`, join, then continue the chain. No graph abstraction.
# ---------------------------------------------------------------------------


class _SleepyTool:
    """Tool that sleeps briefly and records its own start/end times, so a
    test can prove two steps in a parallel group actually overlap instead
    of running one after another."""

    def __init__(self, name: str, delay: float, result: str) -> None:
        from swiftagentx import ToolOutput, ToolOutputType
        self._ToolOutput = ToolOutput
        self.name = name
        self.description = "sleeps then returns a fixed result"
        self.category = "test"
        self.output_type = ToolOutputType.LLM_PROCESSED
        self.timeout_seconds = 5
        self.max_retries = 1
        self._delay = delay
        self._result = result
        self.started_at: float = 0.0
        self.finished_at: float = 0.0

    def get_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def validate_input(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, context: Any, **kwargs: Any):
        import time

        self.started_at = time.monotonic()
        await asyncio.sleep(self._delay)
        self.finished_at = time.monotonic()
        return self._ToolOutput(success=True, result=self._result)


class _KwargsRecordingTool:
    """A Tool that records every kwargs dict it was called with."""

    def __init__(self, name: str) -> None:
        from swiftagentx import ToolOutput, ToolOutputType
        self._ToolOutput = ToolOutput
        self.name = name
        self.description = "records kwargs"
        self.category = "test"
        self.output_type = ToolOutputType.LLM_PROCESSED
        self.timeout_seconds = 5
        self.max_retries = 1
        self.invocations: list[dict[str, Any]] = []

    def get_schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": {}}}

    def validate_input(self, **kwargs: Any) -> bool:
        return True

    async def execute(self, context: Any, **kwargs: Any):
        self.invocations.append(dict(kwargs))
        return self._ToolOutput(success=True, result="ok")


def _direct_scenario_env() -> tuple[Any, Any, Any]:
    """Bare ToolRegistry + ToolExecutor + SessionContext, no Agent — D1
    lives entirely inside `ScenarioEngine.execute_config`, so exercise the
    engine directly instead of round-tripping through a full Agent."""
    from swiftagentx import SessionContext, ToolExecutor, ToolRegistry

    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    context = SessionContext(session_id="s1", user_id="u1", user_input="go")
    return registry, executor, context


@pytest.mark.asyncio
async def test_parallel_group_runs_steps_concurrently() -> None:
    """A parallel group `[stepA, stepB]` must run stepA and stepB
    concurrently (asyncio.gather), not one after another — their execution
    windows must overlap and total wall time must stay near a single
    step's delay, not the sum of both."""
    from swiftagentx import ScenarioConfig, ScenarioEngine, ToolChainStep

    registry, executor, context = _direct_scenario_env()
    tool_a = _SleepyTool("sleepy_a", delay=0.15, result="A-done")
    tool_b = _SleepyTool("sleepy_b", delay=0.15, result="B-done")
    registry.register(tool_a)  # type: ignore[arg-type]
    registry.register(tool_b)  # type: ignore[arg-type]

    engine = ScenarioEngine()
    scenario = ScenarioConfig(
        name="fan_out", triggers=[],
        tool_chain=[[
            ToolChainStep(tool="sleepy_a"),
            ToolChainStep(tool="sleepy_b"),
        ]],
    )

    start = time.monotonic()
    result = await engine.execute_config(scenario, "fan_out", context, executor)
    elapsed = time.monotonic() - start

    assert result.success
    # Sequential execution would take >= 0.3s; concurrent stays well under it.
    assert elapsed < 0.28, f"steps did not run concurrently, took {elapsed:.3f}s"
    assert tool_a.started_at < tool_b.finished_at
    assert tool_b.started_at < tool_a.finished_at


@pytest.mark.asyncio
async def test_parallel_group_merges_results_and_injects_variables() -> None:
    """Each step's `extract_to` result must land in the shared variable bag
    so a later serial step can reference both via `$var` templates."""
    from swiftagentx import ScenarioConfig, ScenarioEngine, ToolChainStep

    registry, executor, context = _direct_scenario_env()
    registry.register(_SleepyTool("fetch_price", delay=0.01, result="42"))  # type: ignore[arg-type]
    registry.register(_SleepyTool("fetch_stock", delay=0.01, result="in_stock"))  # type: ignore[arg-type]
    combine = _KwargsRecordingTool("combine")
    registry.register(combine)  # type: ignore[arg-type]

    engine = ScenarioEngine()
    scenario = ScenarioConfig(
        name="fan_out_join", triggers=[],
        tool_chain=[
            [
                ToolChainStep(tool="fetch_price", extract_to="price"),
                ToolChainStep(tool="fetch_stock", extract_to="stock"),
            ],
            ToolChainStep(tool="combine",
                          kwargs_template={"price": "$price", "stock": "$stock"}),
        ],
        output_type="direct",
    )

    result = await engine.execute_config(scenario, "fan_out_join", context, executor)

    assert result.success
    assert combine.invocations == [{"price": "42", "stock": "in_stock"}]


@pytest.mark.asyncio
async def test_parallel_group_failure_stops_chain() -> None:
    """If any step in a parallel group fails, the scenario stops after the
    group joins — same default behaviour as a failing serial step. A
    configurable fail_fast/best_effort policy is scoped to D2."""
    from swiftagentx import ScenarioConfig, ScenarioEngine, ToolChainStep, ToolOutput, ToolOutputType

    registry, executor, context = _direct_scenario_env()

    class _FailingTool:
        name = "flaky"
        description = "always fails"
        category = "test"
        output_type = ToolOutputType.LLM_PROCESSED
        timeout_seconds = 5
        max_retries = 1

        def get_schema(self) -> dict[str, Any]:
            return {"name": self.name, "description": self.description,
                    "parameters": {"type": "object", "properties": {}}}

        def validate_input(self, **kwargs: Any) -> bool:
            return True

        async def execute(self, context: Any, **kwargs: Any) -> Any:
            return ToolOutput(success=False, result=None, error="boom")

    after_tool = _KwargsRecordingTool("after")
    registry.register(_SleepyTool("ok_step", delay=0.01, result="fine"))  # type: ignore[arg-type]
    registry.register(_FailingTool())  # type: ignore[arg-type]
    registry.register(after_tool)  # type: ignore[arg-type]

    engine = ScenarioEngine()
    scenario = ScenarioConfig(
        name="fan_out_fail", triggers=[],
        tool_chain=[
            [ToolChainStep(tool="ok_step"), ToolChainStep(tool="flaky")],
            ToolChainStep(tool="after"),
        ],
    )

    result = await engine.execute_config(scenario, "fan_out_fail", context, executor)

    assert result.success is False
    assert after_tool.invocations == [], "chain must not continue past a failed group"


# ---------------------------------------------------------------------------
# Friction #9 — Middleware chain must actually execute around run()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_middleware_wraps_run_in_order() -> None:
    """`agent.use(LoggingMiddleware())` followed by `agent.run(...)` should
    invoke middleware.process before and after the request body."""
    from swiftagentx import Middleware

    seen: list[str] = []

    class FirstMW(Middleware):
        async def process(self, context: Any, next_handler: Any) -> Any:
            seen.append("first.before")
            result = await next_handler(context)
            seen.append("first.after")
            return result

    class SecondMW(Middleware):
        async def process(self, context: Any, next_handler: Any) -> Any:
            seen.append("second.before")
            result = await next_handler(context)
            seen.append("second.after")
            return result

    agent = Agent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False),
    )
    agent.use(FirstMW())
    agent.use(SecondMW())

    response = await agent.run("hello")
    assert response.answer  # ran the body
    assert seen == [
        "first.before", "second.before",
        "second.after", "first.after",
    ]


@pytest.mark.asyncio
async def test_no_middleware_takes_fast_path() -> None:
    """When no middleware is registered, the chain isn't invoked and run()
    behaves identically to v0.3.0."""
    agent = Agent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False),
    )
    response = await agent.run("hello")
    assert response.answer
    # Doesn't fail — no middleware, just the bare path.


@pytest.mark.asyncio
async def test_middleware_can_short_circuit_request() -> None:
    """A middleware that doesn't call next_handler should prevent the
    inner work from running."""
    from swiftagentx import Middleware

    class ShortCircuitMW(Middleware):
        async def process(self, context: Any, next_handler: Any) -> Any:
            # Don't call next_handler; produce a synthetic response.
            from swiftagentx.models.schema import AgentResponse
            context["response"] = AgentResponse(
                session_id="x", request_id="r",
                answer="blocked by middleware",
                total_iterations=0, execution_time_ms=0.0,
            )
            return context

    body_ran = {"ran": False}

    class SignalAgent(Agent):
        async def _run_internal(self, **kw: Any) -> Any:  # type: ignore[override]
            body_ran["ran"] = True
            return await super()._run_internal(**kw)

    agent = SignalAgent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False),
    )
    agent.use(ShortCircuitMW())

    response = await agent.run("hello")
    assert response.answer == "blocked by middleware"
    assert body_ran["ran"] is False


# ---------------------------------------------------------------------------
# Friction #10 — FastAPI admin router must NOT double-prefix
# ---------------------------------------------------------------------------


def test_fastapi_admin_router_mounts_at_user_prefix() -> None:
    """The README says:
        app.include_router(create_fastapi_admin_router(svc), prefix="/admin")
    Pre-v0.3.x this gave /admin/admin/status. Post-fix it gives /admin/status.
    """
    fastapi = pytest.importorskip("fastapi")
    starlette_testclient = pytest.importorskip("starlette.testclient")

    from swiftagentx.admin import AdminService, create_fastapi_admin_router

    agent = Agent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False),
    )
    svc = AdminService(agent)

    app = fastapi.FastAPI()
    app.include_router(create_fastapi_admin_router(svc), prefix="/admin")

    client = starlette_testclient.TestClient(app)

    # Correct path returns 200.
    r = client.get("/admin/status")
    assert r.status_code == 200, f"expected 200 at /admin/status, got {r.status_code}: {r.text[:200]}"
    body = r.json()
    assert "name" in body or "agent" in body or "tool_count" in body

    # Wrong (legacy double-prefix) path returns 404.
    r2 = client.get("/admin/admin/status")
    assert r2.status_code == 404


# ---------------------------------------------------------------------------
# Contract test — every lifecycle HookEvent must be dispatched somewhere
# in run() or run_stream(). This stops the recurring "declared but
# never wired" pattern (pipeline → middleware → tool/scenario/react hooks).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_lifecycle_hook_event_fires_at_least_once() -> None:
    """For every HookEvent.lifecycle_events(), assert that registering a
    hook on it and running a request that takes the ReAct + Scenario
    branches actually invokes the handler.

    The previous bug: BEFORE_TOOL_CALL / AFTER_TOOL_CALL /
    BEFORE_SCENARIO_STEP / AFTER_SCENARIO_STEP / BEFORE_REACT_ITER /
    AFTER_REACT_ITER were in the enum but no agent code dispatched them.
    """
    from swiftagentx import (
        HookContext,
        HookEvent,
        HookResult,
        PythonHook,
        ScenarioConfig,
        Tool,
        ToolChainStep,
        ToolOutput,
    )

    fired: set[HookEvent] = set()

    async def handler(ctx: HookContext) -> HookResult:
        fired.add(ctx.event)
        return HookResult()

    # ReAct path agent — forces REACT level, proposes a tool call, then a
    # final answer (so the ReAct iter & tool hooks all fire).
    class _ReActStubAgent(Agent):
        _step = 0

        async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
            from swiftagentx.core.router import IntentLevel, IntentResult
            return IntentResult(level=IntentLevel.REACT, confidence=1.0)

        async def _generate_thought(self, context: Any, model: Any, accumulated: str) -> str:  # type: ignore[override]
            _ReActStubAgent._step += 1
            if _ReActStubAgent._step == 1:
                return 'Thought: I need a tool.\nAction: ping\nAction Input: {"q": "1"}'
            return "Final Answer: done"

    class _PingTool(Tool):
        def __init__(self) -> None:
            super().__init__(name="ping", description="ping back")

        async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
            return ToolOutput(success=True, result="pong")

    agent_react = _ReActStubAgent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False, max_iterations=4),
    )
    agent_react.tool_registry.register(_PingTool())

    # Register one hook for every lifecycle event.
    for event in HookEvent.lifecycle_events():
        agent_react.hooks.register(PythonHook(
            name=f"watch-{event.value}", events={event}, handler=handler,
        ))

    await agent_react.run("anything to trigger react")

    react_events = {
        HookEvent.SESSION_START, HookEvent.REQUEST_START,
        HookEvent.BEFORE_CLASSIFY, HookEvent.AFTER_CLASSIFY,
        HookEvent.BEFORE_REACT_ITER, HookEvent.AFTER_REACT_ITER,
        HookEvent.BEFORE_TOOL_CALL, HookEvent.AFTER_TOOL_CALL,
        HookEvent.BEFORE_RESPOND, HookEvent.REQUEST_END,
    }
    missing_react = react_events - fired
    assert not missing_react, (
        f"ReAct path did not dispatch these lifecycle events: {missing_react}. "
        f"All declared HookEvent.lifecycle_events() must fire when a real "
        f"request takes the relevant code path."
    )

    # Scenario path agent — forces SCENARIO level for the BEFORE/AFTER_SCENARIO_STEP events.
    class _ScenarioStubAgent(Agent):
        async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
            from swiftagentx.core.router import IntentLevel, IntentResult
            return IntentResult(level=IntentLevel.SCENARIO, scenario="weather", confidence=1.0)

    fired.clear()
    agent_scn = _ScenarioStubAgent(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False),
    )
    agent_scn.tool_registry.register(_PingTool())
    agent_scn.register_scenario("weather", ScenarioConfig(
        name="w", description="", triggers=["weather"],
        tool_chain=[ToolChainStep(tool="ping", query_template="x")],
        cache_ttl=60, output_type="direct",
    ))
    for event in HookEvent.lifecycle_events():
        agent_scn.hooks.register(PythonHook(
            name=f"watch-{event.value}", events={event}, handler=handler,
        ))

    await agent_scn.run("anything to trigger scenario")
    assert HookEvent.BEFORE_SCENARIO_STEP in fired, "BEFORE_SCENARIO_STEP never fired"
    assert HookEvent.AFTER_SCENARIO_STEP in fired, "AFTER_SCENARIO_STEP never fired"


# ---------------------------------------------------------------------------
# Friction #11 (this round) — ScenarioEngine supports multi-kwarg tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scenario_kwargs_template_for_multi_param_tool() -> None:
    """An MCP-style tool taking multiple kwargs (a, b) should be drivable
    from a Scenario via `kwargs_template={"a": "$x", "b": "$y"}`."""
    from swiftagentx import (
        ScenarioConfig,
        Tool,
        ToolChainStep,
        ToolOutput,
    )

    captured: list[dict[str, Any]] = []

    class _AddTool(Tool):
        def __init__(self) -> None:
            super().__init__(name="add", description="add a + b")

        async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
            captured.append(dict(kwargs))
            try:
                total = int(kwargs["a"]) + int(kwargs["b"])
            except Exception as exc:
                return ToolOutput(success=False, result=None, error=str(exc))
            return ToolOutput(success=True, result=str(total))

    class _ForceScenario(Agent):
        async def _classify_intent(self, user_input: str, context: Any):  # type: ignore[override]
            from swiftagentx.core.router import IntentLevel, IntentResult
            return IntentResult(level=IntentLevel.SCENARIO, scenario="math",
                                confidence=1.0)

    agent = _ForceScenario(
        model=DummyModelClient(api_key="k", model="d"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False),
    )
    agent.tool_registry.register(_AddTool())
    agent.register_scenario("math", ScenarioConfig(
        name="math", description="", triggers=["sum"],
        tool_chain=[ToolChainStep(
            tool="add",
            kwargs_template={"a": "$lhs", "b": "$rhs"},
        )],
        cache_ttl=60,
        output_type="direct",
    ))

    response = await agent.run("compute the sum", lhs="3", rhs="4")
    assert captured == [{"a": "3", "b": "4"}], f"got {captured}"
    assert "7" in response.answer


# ---------------------------------------------------------------------------
# Friction D-5 — run_stream survives a disconnected consumer fast
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stream_with_disconnected_consumer_finishes_fast() -> None:
    """If the SSE consumer disappears (buffer fills + nobody drains it),
    the producer should mark the adapter closed within ~1s and silently
    drop further events, instead of blocking 5s per send_event and
    raising a naked RuntimeError out of run_stream."""
    import time as _time

    from swiftagentx import AgentRequest, SSEStreamAdapter
    from swiftagentx.core.model_client import ModelClient, ModelResponse

    class FloodingModel(ModelClient):
        async def chat(self, messages: Any, **kw: Any) -> Any:
            return ModelResponse(content="x" * 200, model=self.model)

        async def complete(self, prompt: Any, **kw: Any) -> Any:
            return ModelResponse(content="x" * 200, model=self.model)

        async def stream_chat(self, messages: Any, **kw: Any) -> Any:
            for _ in range(200):
                yield "x"

        async def stream_complete(self, prompt: Any, **kw: Any) -> Any:
            for ch in "x" * 200:
                yield ch

    agent = Agent(
        model=FloodingModel(api_key="k", model="m"),
        config=SwiftAgentConfig(memory_enable_topic_change_hook=False,
                                enable_cache=False),
    )
    adapter = SSEStreamAdapter(buffer_size=5, put_timeout=0.1)
    request = AgentRequest(user_id="u", session_id="s", user_input="hello")

    start = _time.perf_counter()
    response = await agent.run_stream(request, adapter)
    elapsed = _time.perf_counter() - start

    # Must finish well under the old 10s pathology.
    assert elapsed < 2.0, (
        f"run_stream took {elapsed:.1f}s with disconnected consumer; "
        f"expected <2.0s after the v0.3.1 disconnect fix."
    )
    assert response.answer  # returned cleanly, didn't raise
    assert adapter.is_closed or adapter.is_finished
    assert adapter.events_dropped >= 1
