"""Tests for the Planner fast path (core/planner.py + agent wiring)."""

import pytest

from swiftagentx import Agent
from swiftagentx.core.model_client import DummyModelClient, ModelResponse
from swiftagentx.core.planner import GeneratedPlan, Planner, PlanStore
from swiftagentx.core.router import IntentLevel, IntentResult
from swiftagentx.models.config import ModelTier, SwiftAgentConfig
from swiftagentx.tools.base import Tool, ToolOutput
from swiftagentx.tools.scenario import ToolChainStep

PLAN_JSON = (
    '{"intent": "weather_then_calc", "description": "查天气再计算",'
    ' "steps": ['
    '{"tool": "weather", "kwargs": {"city": "$city"}, "extract_to": "weather_info"},'
    '{"tool": "calc", "kwargs": {"expression": "$expr"}}],'
    ' "slots": {"city": "北京", "expr": "1+1"}}'
)


# ---------------------------------------------------------------------------
# Planner.parse / validate
# ---------------------------------------------------------------------------

class TestPlannerParse:
    def test_valid_plan(self):
        plan = Planner().parse(PLAN_JSON)
        assert plan is not None
        assert plan.intent == "weather_then_calc"
        assert [s.tool for s in plan.steps] == ["weather", "calc"]
        assert plan.steps[0].kwargs_template == {"city": "$city"}
        assert plan.steps[0].extract_to == "weather_info"
        assert plan.slots == {"city": "北京", "expr": "1+1"}

    def test_not_plannable_reply(self):
        assert Planner().parse('{"intent": null}') is None

    def test_garbage(self):
        assert Planner().parse("I think we should use a tool maybe?") is None

    def test_fenced_json(self):
        plan = Planner().parse(f"Sure!\n```json\n{PLAN_JSON}\n```\n")
        assert plan is not None and plan.intent == "weather_then_calc"

    def test_intent_sanitized(self):
        plan = Planner().parse(
            '{"intent": "Weather & Calc!", "steps": [{"tool": "t", "kwargs": {}}]}'
        )
        assert plan is not None
        assert plan.intent == "weather___calc"


class TestPlannerValidate:
    def _plan(self, **kwargs) -> GeneratedPlan:
        base = dict(
            intent="p",
            steps=[ToolChainStep(tool="weather", kwargs_template={"city": "$city"})],
            slots={"city": "北京"},
        )
        base.update(kwargs)
        return GeneratedPlan(**base)

    def test_valid(self):
        assert Planner().validate(self._plan(), {"weather"}) is None

    def test_unknown_tool(self):
        err = Planner().validate(self._plan(), {"calc"})
        assert err is not None and "unknown tool" in err

    def test_undefined_var(self):
        plan = self._plan(slots={})
        err = Planner().validate(plan, {"weather"})
        assert err is not None and "$city" in err

    def test_var_from_earlier_extract_to_ok(self):
        plan = GeneratedPlan(intent="p", slots={"city": "北京"}, steps=[
            ToolChainStep(tool="weather", kwargs_template={"city": "$city"},
                          extract_to="info"),
            ToolChainStep(tool="notify", kwargs_template={"body": "$info"}),
        ])
        assert Planner().validate(plan, {"weather", "notify"}) is None

    def test_var_from_later_step_rejected(self):
        plan = GeneratedPlan(intent="p", slots={}, steps=[
            ToolChainStep(tool="notify", kwargs_template={"body": "$info"}),
            ToolChainStep(tool="weather", kwargs_template={}, extract_to="info"),
        ])
        assert Planner().validate(plan, {"weather", "notify"}) is not None

    def test_too_many_steps(self):
        plan = self._plan(steps=[
            ToolChainStep(tool="weather", kwargs_template={}) for _ in range(6)
        ])
        err = Planner(max_steps=5).validate(plan, {"weather"})
        assert err is not None and "max 5" in err

    def test_empty_steps(self):
        assert Planner().validate(self._plan(steps=[]), {"weather"}) is not None

    def test_reserved_vars_allowed(self):
        plan = GeneratedPlan(intent="p", slots={}, steps=[
            ToolChainStep(tool="search", kwargs_template={"q": "$user_input"}),
        ])
        assert Planner().validate(plan, {"search"}) is None


# ---------------------------------------------------------------------------
# PlanStore
# ---------------------------------------------------------------------------

def _gen_plan() -> GeneratedPlan:
    return Planner().parse(PLAN_JSON)


class TestPlanStore:
    def test_match_same_template_different_slot_values(self):
        store = PlanStore()
        store.add(_gen_plan(), "帮我查北京天气然后算1+1")
        assert store.match("帮我查上海天气然后算2*3") is not None

    def test_no_match_for_unrelated_query(self):
        store = PlanStore()
        store.add(_gen_plan(), "帮我查北京天气然后算1+1")
        assert store.match("给我订一张去伦敦的机票") is None

    def test_same_shape_dedupes(self):
        store = PlanStore()
        a = store.add(_gen_plan(), "查北京天气算1+1")
        b = store.add(_gen_plan(), "查上海天气算2+2")
        assert a.plan_id == b.plan_id
        assert len(store.list_plans()) == 1
        assert len(a.source_queries) == 2

    def test_rule_track_promotable_after_n_successes(self):
        store = PlanStore(promote_after=3)
        cached = store.add(_gen_plan(), "查北京天气算1+1")
        assert store.record_success(cached.plan_id) is False
        assert store.record_success(cached.plan_id) is False
        assert store.record_success(cached.plan_id) is True

    def test_failure_blocks_promotion_and_evicts(self):
        store = PlanStore(promote_after=2, evict_after_failures=2)
        cached = store.add(_gen_plan(), "查北京天气算1+1")
        store.record_failure(cached.plan_id)
        assert store.record_success(cached.plan_id) is False  # failures > 0
        store.record_failure(cached.plan_id)
        assert store.get(cached.plan_id) is None  # evicted

    def test_promoted_plan_excluded_from_match(self):
        store = PlanStore()
        cached = store.add(_gen_plan(), "帮我查北京天气然后算1+1")
        store.mark_promoted(cached.plan_id)
        assert store.match("帮我查上海天气然后算2+2") is None

    def test_to_scenario_config(self):
        store = PlanStore()
        cached = store.add(_gen_plan(), "查北京天气算1+1")
        config = store.to_scenario_config(cached.plan_id)
        assert config is not None
        assert config.name == "weather_then_calc"
        assert config.triggers == ["查北京天气算1+1"]
        assert [s.tool for s in config.tool_chain] == ["weather", "calc"]
        assert config.required_vars() == {"city", "expr"}


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

class _ScriptedModel(DummyModelClient):
    """Routes each framework prompt to a canned reply by prompt markers."""

    def __init__(self, plan_reply: str = PLAN_JSON):
        super().__init__(api_key="k", model="scripted")
        self.plan_reply = plan_reply
        self.planner_calls = 0
        self.extract_calls = 0

    async def chat(self, messages, **kwargs) -> ModelResponse:
        prompt = messages[-1]["content"]
        if "tool-call planner" in prompt:
            self.planner_calls += 1
            return ModelResponse(content=self.plan_reply, model="scripted")
        if "Extract these slot values" in prompt:
            self.extract_calls += 1
            return ModelResponse(
                content='slots={"city": "上海", "expr": "2*3"}', model="scripted",
            )
        return ModelResponse(content="synthesized answer", model="scripted")


class _WeatherTool(Tool):
    def __init__(self, calls: list):
        super().__init__(name="weather", description="query city weather")
        self.calls = calls

    async def execute(self, context, **kwargs) -> ToolOutput:
        self.calls.append(("weather", kwargs))
        return ToolOutput(success=True, result=f"晴 25C ({kwargs.get('city')})")


class _CalcTool(Tool):
    def __init__(self, calls: list, fail: bool = False):
        super().__init__(name="calc", description="evaluate arithmetic")
        self.calls = calls
        self.fail = fail

    async def execute(self, context, **kwargs) -> ToolOutput:
        self.calls.append(("calc", kwargs))
        if self.fail:
            return ToolOutput(success=False, result=None, error="boom")
        return ToolOutput(success=True, result="2")


class _ForcedReactAgent(Agent):
    """Always classifies REACT; ReAct loop is a visible stub."""

    react_entered = 0

    async def _classify_intent(self, user_input, context):  # type: ignore[override]
        return IntentResult(level=IntentLevel.REACT, confidence=1.0)

    async def _react_loop(self, context, adapter=None):  # type: ignore[override]
        type(self).react_entered += 1
        return "react fallback answer"


def _make_agent(model, fail_calc: bool = False, **config_overrides):
    calls: list = []
    config_kwargs = {
        "enable_planner": True, "enable_cache": False,
        "memory_enable_topic_change_hook": False, **config_overrides,
    }
    agent = _ForcedReactAgent(
        models={ModelTier.LIGHT: model, ModelTier.HEAVY: model},
        config=SwiftAgentConfig(**config_kwargs),
    )
    agent.tool_registry.register(_WeatherTool(calls))
    agent.tool_registry.register(_CalcTool(calls, fail=fail_calc))
    _ForcedReactAgent.react_entered = 0
    return agent, calls


class TestAgentPlannerIntegration:
    @pytest.mark.asyncio
    async def test_fresh_plan_executes_tools_in_order(self):
        model = _ScriptedModel()
        agent, calls = _make_agent(model)
        response = await agent.run("帮我查北京天气然后算1+1")

        assert response.answer == "synthesized answer"
        assert [c[0] for c in calls] == ["weather", "calc"]
        assert calls[0][1] == {"city": "北京"}
        assert calls[1][1] == {"expression": "1+1"}
        assert _ForcedReactAgent.react_entered == 0
        assert len(agent.plan_store.list_plans()) == 1

    @pytest.mark.asyncio
    async def test_cached_plan_reused_with_new_slots(self):
        model = _ScriptedModel()
        agent, calls = _make_agent(model)
        await agent.run("帮我查北京天气然后算1+1")
        response = await agent.run("帮我查上海天气然后算2*3")

        assert response.answer == "synthesized answer"
        assert model.planner_calls == 1      # second run reused the cache
        assert model.extract_calls == 1
        assert calls[2][1] == {"city": "上海"}
        assert calls[3][1] == {"expression": "2*3"}
        assert _ForcedReactAgent.react_entered == 0

    @pytest.mark.asyncio
    async def test_step_failure_falls_back_to_react(self):
        model = _ScriptedModel()
        agent, calls = _make_agent(model, fail_calc=True)
        response = await agent.run("帮我查北京天气然后算1+1")

        assert response.answer == "react fallback answer"
        assert _ForcedReactAgent.react_entered == 1
        # A plan that failed execution must not enter the cache.
        assert agent.plan_store.list_plans() == []

    @pytest.mark.asyncio
    async def test_invalid_plan_falls_back_without_running_tools(self):
        model = _ScriptedModel(plan_reply=(
            '{"intent": "x", "steps": [{"tool": "ghost", "kwargs": {}}], "slots": {}}'
        ))
        agent, calls = _make_agent(model)
        response = await agent.run("做点什么")

        assert response.answer == "react fallback answer"
        assert calls == []

    @pytest.mark.asyncio
    async def test_planner_disabled_goes_straight_to_react(self):
        model = _ScriptedModel()
        agent, calls = _make_agent(model, enable_planner=False)
        response = await agent.run("帮我查北京天气然后算1+1")

        assert response.answer == "react fallback answer"
        assert model.planner_calls == 0
        assert calls == []

    @pytest.mark.asyncio
    async def test_rule_track_auto_promotes_to_scenario(self):
        model = _ScriptedModel()
        agent, _ = _make_agent(model, plan_promote_after=2, plan_auto_promote=True)
        await agent.run("帮我查北京天气然后算1+1")
        await agent.run("帮我查上海天气然后算2*3")

        plan = agent.plan_store.list_plans()[0]
        assert plan.promoted is True
        assert agent.scenario_engine.get(plan.plan_id) is not None
        assert plan.plan_id in agent.router._scenarios
        # Real user phrasings became the scenario's retrieval triggers.
        assert agent.router._scenarios[plan.plan_id]["triggers"]

    @pytest.mark.asyncio
    async def test_default_promotion_is_manual(self):
        # plan_auto_promote defaults to False: plans accumulate successes
        # but are never registered behind the developer's back.
        model = _ScriptedModel()
        agent, _ = _make_agent(model, plan_promote_after=1)
        await agent.run("帮我查北京天气然后算1+1")

        plan = agent.plan_store.list_plans()[0]
        assert plan.successes >= 1
        assert plan.promoted is False
        assert agent.scenario_engine.get(plan.plan_id) is None

    @pytest.mark.asyncio
    async def test_manual_track_when_auto_promote_off(self):
        model = _ScriptedModel()
        agent, _ = _make_agent(model, plan_promote_after=1, plan_auto_promote=False)
        await agent.run("帮我查北京天气然后算1+1")

        plan = agent.plan_store.list_plans()[0]
        assert plan.promoted is False
        assert agent.scenario_engine.get(plan.plan_id) is None

        # export gives a reviewable draft; promote registers it.
        draft = agent.export_plan_scenario(plan.plan_id)
        assert draft is not None and draft.name == "weather_then_calc"
        assert agent.promote_plan(plan.plan_id) is True
        assert agent.scenario_engine.get(plan.plan_id) is not None
        assert agent.list_plan_candidates()[0]["promoted"] is True
