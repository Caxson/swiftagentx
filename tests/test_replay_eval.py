"""Tests for D6 replay eval gate (core/replay_eval.py + agent wiring)."""

import pytest

from swiftagentx import Agent, DummyModelClient, InMemoryWorkspaceBackend, SessionContext
from swiftagentx.core.miner import ReactTranscript
from swiftagentx.core.planner import CachedPlan, GeneratedPlan
from swiftagentx.core.replay_eval import (
    ReplayEvaluator,
    agreement_score,
    save_report,
    select_cases,
)
from swiftagentx.core.router import IntentLevel, IntentResult
from swiftagentx.models.config import ModelTier, SwiftAgentConfig
from swiftagentx.tools.base import Tool, ToolOutput
from swiftagentx.tools.executor import ToolExecutor
from swiftagentx.tools.registry import ToolRegistry
from swiftagentx.tools.scenario import ToolChainStep


def _plan(**overrides) -> CachedPlan:
    defaults = dict(
        plan_id="plan_weather_calc_abcd1234",
        intent="weather_calc",
        steps=[
            ToolChainStep(tool="weather", kwargs_template={"city": "$city"}),
            ToolChainStep(tool="calc", kwargs_template={"expression": "$expression"}),
        ],
    )
    defaults.update(overrides)
    return CachedPlan(**defaults)


def _transcript(user_input: str, city: str, expr: str, baseline_output: str) -> ReactTranscript:
    return ReactTranscript(
        user_input=user_input,
        actions=[("weather", {"city": city}), ("calc", {"expression": expr})],
        baseline_output=baseline_output,
    )


class _WeatherTool(Tool):
    def __init__(self, result: str = "sunny 25C"):
        super().__init__(name="weather", description="query city weather")
        self._result = result

    async def execute(self, context, **kwargs) -> ToolOutput:
        return ToolOutput(success=True, result=self._result)


class _CalcTool(Tool):
    def __init__(self, result: str = "2"):
        super().__init__(name="calc", description="evaluate arithmetic")
        self._result = result

    async def execute(self, context, **kwargs) -> ToolOutput:
        return ToolOutput(success=True, result=self._result)


class _FailingTool(Tool):
    def __init__(self):
        super().__init__(name="calc", description="always fails")

    async def execute(self, context, **kwargs) -> ToolOutput:
        return ToolOutput(success=False, result=None, error="boom")


def _env(calc_result: str = "sunny 25C, 2") -> tuple[ToolExecutor, SessionContext]:
    registry = ToolRegistry()
    registry.register(_WeatherTool())
    registry.register(_CalcTool(result=calc_result))
    executor = ToolExecutor(registry)
    context = SessionContext(session_id="eval", user_id="eval", user_input="")
    return executor, context


# ---------------------------------------------------------------------------
# agreement_score
# ---------------------------------------------------------------------------

class TestAgreementScore:
    def test_identical_strings_score_one(self):
        assert agreement_score("sunny 25C", "sunny 25C") == 1.0

    def test_disjoint_strings_score_zero(self):
        assert agreement_score("sunny 25C", "rainy -3C") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        score = agreement_score("北京晴天25度", "北京晴天20度")
        assert 0.0 < score < 1.0

    def test_both_empty_scores_one(self):
        assert agreement_score("", "") == 1.0

    def test_one_empty_scores_zero(self):
        assert agreement_score("sunny", "") == 0.0
        assert agreement_score("", "sunny") == 0.0


# ---------------------------------------------------------------------------
# select_cases
# ---------------------------------------------------------------------------

class TestSelectCases:
    def test_matches_same_shape_transcripts_only(self):
        plan = _plan()
        matching = _transcript("查北京天气算1+1", "北京", "1+1", "sunny 2")
        other_shape = ReactTranscript(
            user_input="订机票", actions=[("book_flight", {"dest": "伦敦"})],
        )
        cases = select_cases(plan, [matching, other_shape])
        assert cases == [matching]

    def test_no_matches_returns_empty(self):
        plan = _plan()
        other_shape = ReactTranscript(
            user_input="订机票", actions=[("book_flight", {"dest": "伦敦"})],
        )
        assert select_cases(plan, [other_shape]) == []


# ---------------------------------------------------------------------------
# ReplayEvaluator.evaluate — threshold judgment
# ---------------------------------------------------------------------------

class TestReplayEvaluator:
    @pytest.mark.asyncio
    async def test_all_cases_agree_verdict_passes(self):
        executor, context = _env(calc_result="sunny 25C 2")
        plan = _plan()
        cases = [
            _transcript("北京天气加1+1", "北京", "1+1", "sunny 25C 2"),
            _transcript("上海天气加1+1", "上海", "1+1", "sunny 25C 2"),
        ]
        evaluator = ReplayEvaluator(executor, agreement_threshold=0.5, pass_rate_threshold=0.7)
        report = await evaluator.evaluate(plan, cases, context)

        assert report.total_cases == 2
        assert report.passed_cases == 2
        assert report.pass_rate == 1.0
        assert report.verdict is True
        assert all(c.passed for c in report.cases)

    @pytest.mark.asyncio
    async def test_disagreeing_cases_fail_the_gate(self):
        executor, context = _env(calc_result="sunny 25C 2")
        plan = _plan()
        cases = [
            _transcript("北京天气加1+1", "北京", "1+1", "completely different unrelated text"),
        ]
        evaluator = ReplayEvaluator(executor, agreement_threshold=0.5, pass_rate_threshold=0.7)
        report = await evaluator.evaluate(plan, cases, context)

        assert report.total_cases == 1
        assert report.passed_cases == 0
        assert report.verdict is False

    @pytest.mark.asyncio
    async def test_mixed_pass_rate_below_threshold_fails(self):
        executor, context = _env(calc_result="sunny 25C 2")
        plan = _plan()
        cases = [
            _transcript("q1", "北京", "1+1", "sunny 25C 2"),
            _transcript("q2", "上海", "1+1", "totally unrelated garbage"),
        ]
        # 1/2 = 0.5 pass rate, below the 0.7 bar.
        evaluator = ReplayEvaluator(executor, agreement_threshold=0.5, pass_rate_threshold=0.7)
        report = await evaluator.evaluate(plan, cases, context)

        assert report.passed_cases == 1
        assert report.pass_rate == 0.5
        assert report.verdict is False

    @pytest.mark.asyncio
    async def test_mixed_pass_rate_at_lower_threshold_passes(self):
        executor, context = _env(calc_result="sunny 25C 2")
        plan = _plan()
        cases = [
            _transcript("q1", "北京", "1+1", "sunny 25C 2"),
            _transcript("q2", "上海", "1+1", "totally unrelated garbage"),
        ]
        evaluator = ReplayEvaluator(executor, agreement_threshold=0.5, pass_rate_threshold=0.5)
        report = await evaluator.evaluate(plan, cases, context)
        assert report.verdict is True

    @pytest.mark.asyncio
    async def test_no_cases_never_passes(self):
        executor, context = _env()
        plan = _plan()
        evaluator = ReplayEvaluator(executor)
        report = await evaluator.evaluate(plan, [], context)
        assert report.total_cases == 0
        assert report.verdict is False

    @pytest.mark.asyncio
    async def test_failed_tool_call_replays_as_empty_output(self):
        registry = ToolRegistry()
        registry.register(_WeatherTool())
        registry.register(_FailingTool())
        executor = ToolExecutor(registry)
        context = SessionContext(session_id="eval", user_id="eval", user_input="")
        plan = _plan()
        cases = [_transcript("q1", "北京", "1+1", "sunny 25C 2")]
        evaluator = ReplayEvaluator(executor, agreement_threshold=0.5)
        report = await evaluator.evaluate(plan, cases, context)
        assert report.cases[0].replay_output == ""
        assert report.cases[0].passed is False


# ---------------------------------------------------------------------------
# save_report — eval report generation, persisted to disk
# ---------------------------------------------------------------------------

class TestSaveReport:
    @pytest.mark.asyncio
    async def test_report_persisted_and_readable(self):
        backend = InMemoryWorkspaceBackend()
        ws = await backend.open("s1")
        executor, context = _env(calc_result="sunny 25C 2")
        plan = _plan()
        cases = [_transcript("q1", "北京", "1+1", "sunny 25C 2")]
        evaluator = ReplayEvaluator(executor)
        report = await evaluator.evaluate(plan, cases, context)

        await save_report(ws, report)

        raw = await ws.read(f"eval_reports/{plan.plan_id}.json")
        assert raw is not None
        import json
        loaded = json.loads(raw)
        assert loaded["plan_id"] == plan.plan_id
        assert loaded["verdict"] is True
        assert loaded["total_cases"] == 1


# ---------------------------------------------------------------------------
# Agent integration: enable_replay_eval wiring + replay_eval_plan()
# ---------------------------------------------------------------------------

_CHAIN_VALUES = {
    "查北京天气算1+1": ("北京", "1+1"),
    "查上海天气算2+2": ("上海", "2+2"),
    "查广州天气算3+3": ("广州", "3+3"),
}


class _ForcedReactAgent(Agent):
    """Always classifies REACT and scripts a deterministic 2-tool chain,
    keyed off the request text (mirrors test_miner.py's fixture)."""

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
        # The ReAct loop's termination check breaks on a literal "Final
        # Answer:" thought *before* the extraction branch runs, so real
        # runs always synthesize the answer here rather than via
        # `_extract_final_answer`. Return a value deterministically tied to
        # the chain's actual tool output ("2") so the replay-eval agreement
        # test isn't at the mercy of DummyModelClient's canned reply text.
        return "2"


def _make_react_agent(**config_overrides) -> Agent:
    model = DummyModelClient(api_key="k", model="d")
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


class TestAgentReplayEval:
    @pytest.mark.asyncio
    async def test_disabled_by_default_no_eval_transcripts_logged(self):
        agent = _make_react_agent()
        await agent.run("查北京天气算1+1")
        assert agent._eval_transcripts == []

    @pytest.mark.asyncio
    async def test_unknown_plan_id_returns_none(self):
        agent = _make_react_agent(enable_replay_eval=True)
        assert await agent.replay_eval_plan("nope") is None

    @pytest.mark.asyncio
    async def test_too_few_cases_returns_none(self):
        agent = _make_react_agent(enable_replay_eval=True, eval_min_cases=5)
        cached = agent.plan_store.add(
            GeneratedPlan(
                intent="weather_calc",
                steps=[
                    ToolChainStep(tool="weather", kwargs_template={"city": "$city"}),
                    ToolChainStep(tool="calc", kwargs_template={"expression": "$expression"}),
                ],
            ),
            "seed query",
        )
        await agent.run("查北京天气算1+1")
        assert await agent.replay_eval_plan(cached.plan_id) is None

    @pytest.mark.asyncio
    async def test_end_to_end_passing_candidate_auto_approves(self):
        agent = _make_react_agent(
            enable_transcript_mining=True, enable_replay_eval=True,
            mining_min_cluster_size=2, eval_min_cases=1,
            plan_auto_reuse=False,
        )

        await agent.run("查北京天气算1+1")
        await agent.run("查上海天气算2+2")
        touched = agent.mine_scenario_candidates()
        assert len(touched) == 1
        plan_id = touched[0]
        assert agent.plan_store.get(plan_id).approved is False

        report = await agent.replay_eval_plan(plan_id)
        assert report is not None
        assert report.total_cases == 2
        assert report.verdict is True
        assert agent.plan_store.get(plan_id).approved is True

        ws = await agent.workspace_backend.open(f"replay_eval_{plan_id}")
        raw = await ws.read(f"eval_reports/{plan_id}.json")
        assert raw is not None
