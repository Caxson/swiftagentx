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


# ---------------------------------------------------------------------------
# Friction #6 — ReAct loop must not call the same tool with the same args twice
# ---------------------------------------------------------------------------


class _CountingTool:
    """A tool that records invocations. Built to look like a real Tool
    without importing Tool ABC (the agent looks it up by name)."""

    def __init__(self) -> None:
        from swiftagentx import Tool, ToolOutput, ToolOutputType
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
        from swiftagentx import Tool, ToolOutput, ToolOutputType
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
