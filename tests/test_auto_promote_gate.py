"""Tests for D7 auto-promotion gate (core/planner.py + agent wiring).

Ties D6's replay-eval verdict into the existing rule-track promotion
(``plan_auto_promote`` + ``plan_promote_after``): with
``plan_promote_requires_eval`` on, N clean successes are no longer enough
by themselves — the candidate must also have cleared the replay-eval gate
before it is auto-registered as a Scenario. Default off leaves the
existing successes-only rule track (already covered by
``test_planner.py::test_rule_track_auto_promotes_to_scenario``) unchanged.
"""

import pytest

from swiftagentx import Agent, InMemoryWorkspaceBackend
from swiftagentx.core.model_client import DummyModelClient, ModelResponse
from swiftagentx.core.router import IntentLevel, IntentResult
from swiftagentx.models.config import ModelTier, SwiftAgentConfig
from swiftagentx.tools.base import Tool, ToolOutput

_CHAIN_VALUES = {
    "查北京天气算1+1": ("北京", "1+1"),
    "查上海天气算2+2": ("上海", "2+2"),
    "查广州天气算3+3": ("广州", "3+3"),
}


class _WeatherTool(Tool):
    def __init__(self):
        super().__init__(name="weather", description="query city weather")

    async def execute(self, context, **kwargs) -> ToolOutput:
        return ToolOutput(success=True, result=f"sunny ({kwargs.get('city')})")


class _CalcTool(Tool):
    def __init__(self):
        super().__init__(name="calc", description="evaluate arithmetic")

    async def execute(self, context, **kwargs) -> ToolOutput:
        return ToolOutput(success=True, result="2")


class _ScriptedExtractModel(DummyModelClient):
    """Scripts slot extraction for the one re-match request each test
    issues after promotion is (or isn't) supposed to happen; everything
    else falls through to the dummy default (e.g. fresh-plan generation,
    which must fail so requests keep flowing through the scripted ReAct
    path instead of a real Planner-generated plan)."""

    async def chat(self, messages, **kwargs):
        prompt = messages[-1]["content"] if messages else ""
        if "Extract these slot values" in prompt:
            city, expr = _CHAIN_VALUES["查广州天气算3+3"]
            return ModelResponse(
                content=f'slots={{"city": "{city}", "expression": "{expr}"}}',
                model="scripted",
            )
        return await super().chat(messages, **kwargs)


class _ForcedReactAgent(Agent):
    """Always classifies REACT and scripts a deterministic 2-tool chain,
    keyed off the request text (mirrors test_replay_eval.py's fixture)."""

    async def _classify_intent(self, user_input, context):  # type: ignore[override]
        return IntentResult(level=IntentLevel.REACT, confidence=1.0)

    async def _generate_thought(self, context, model, accumulated_context):  # type: ignore[override]
        city, expr = _CHAIN_VALUES[context.user_input]
        if context.current_iteration == 1:
            return f'Thought: need weather.\nAction: weather\nAction Input: {{"city": "{city}"}}'
        if context.current_iteration == 2:
            return f'Thought: now calc.\nAction: calc\nAction Input: {{"expression": "{expr}"}}'
        return "Final Answer: 2"

    async def _generate_final_answer(self, context, model, accumulated_context):  # type: ignore[override]
        return "2"


def _make_agent(**config_overrides) -> Agent:
    model = _ScriptedExtractModel(api_key="k", model="d")
    config_kwargs = {
        "enable_cache": False, "memory_enable_topic_change_hook": False,
        **config_overrides,
    }
    agent = _ForcedReactAgent(
        models={ModelTier.LIGHT: model, ModelTier.HEAVY: model},
        config=SwiftAgentConfig(**config_kwargs),
    )
    agent.tool_registry.register(_WeatherTool())
    agent.tool_registry.register(_CalcTool())
    agent.workspace_backend = InMemoryWorkspaceBackend()
    return agent


class TestAutoPromoteGateRequiresEval:
    @pytest.mark.asyncio
    async def test_successes_alone_do_not_promote_when_eval_required(self):
        agent = _make_agent(
            enable_planner=True, enable_transcript_mining=True, enable_replay_eval=True,
            mining_min_cluster_size=2, eval_min_cases=1,
            plan_auto_promote=True, plan_promote_after=1,
            plan_promote_requires_eval=True,
        )
        await agent.run("查北京天气算1+1")
        await agent.run("查上海天气算2+2")
        touched = agent.mine_scenario_candidates()
        plan_id = touched[0]
        # Reuse gate opened manually, WITHOUT ever passing replay eval.
        agent.plan_store.approve(plan_id)

        await agent.run("查广州天气算3+3")  # matches + executes via planner fast path

        plan = agent.plan_store.get(plan_id)
        assert plan.successes >= 1
        assert plan.eval_passed is False
        assert plan.promoted is False
        assert agent.scenario_engine.get(plan_id) is None

    @pytest.mark.asyncio
    async def test_end_to_end_candidate_eval_auto_promote_scenario_hit(self):
        agent = _make_agent(
            enable_planner=True, enable_transcript_mining=True, enable_replay_eval=True,
            mining_min_cluster_size=2, eval_min_cases=1,
            plan_auto_promote=True, plan_promote_after=1,
            plan_promote_requires_eval=True,
        )

        # Candidate: two ReAct runs mined into one plan_store entry.
        await agent.run("查北京天气算1+1")
        await agent.run("查上海天气算2+2")
        touched = agent.mine_scenario_candidates()
        assert len(touched) == 1
        plan_id = touched[0]
        assert agent.plan_store.get(plan_id).approved is False

        # Eval: replay against the mined transcripts clears the gate.
        report = await agent.replay_eval_plan(plan_id)
        assert report is not None and report.verdict is True
        assert agent.plan_store.get(plan_id).approved is True
        assert agent.plan_store.get(plan_id).eval_passed is True

        # Auto-promote: the next matching request re-executes the plan,
        # racks up the required success, and — with eval already
        # passed — is promoted straight to a Scenario, no manual review.
        await agent.run("查广州天气算3+3")

        plan = agent.plan_store.get(plan_id)
        assert plan.promoted is True

        # Scenario hit: the promoted plan is a registered, matchable Scenario.
        assert agent.scenario_engine.get(plan_id) is not None
        assert plan_id in agent.router._scenarios
        assert agent.router._scenarios[plan_id]["triggers"]
