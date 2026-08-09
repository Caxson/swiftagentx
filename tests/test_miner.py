"""Tests for D5 transcript mining (core/miner.py + agent wiring)."""

import pytest

from swiftagentx import Agent
from swiftagentx.core.miner import ReactTranscript, TranscriptMiner
from swiftagentx.core.model_client import DummyModelClient
from swiftagentx.core.planner import PlanStore
from swiftagentx.core.router import IntentLevel, IntentResult
from swiftagentx.models.config import ModelTier, SwiftAgentConfig
from swiftagentx.tools.base import Tool, ToolOutput


def _transcript(user_input: str, city: str, expr: str) -> ReactTranscript:
    return ReactTranscript(
        user_input=user_input,
        actions=[
            ("weather", {"city": city}),
            ("calc", {"expression": expr}),
        ],
    )


# ---------------------------------------------------------------------------
# TranscriptMiner (unit)
# ---------------------------------------------------------------------------

class TestTranscriptMiner:
    def test_below_threshold_produces_no_candidate(self):
        miner = TranscriptMiner(min_cluster_size=3)
        store = PlanStore()
        transcripts = [
            _transcript("查北京天气算1+1", "北京", "1+1"),
            _transcript("查上海天气算2+2", "上海", "2+2"),
        ]
        touched = miner.mine(transcripts, store)
        assert touched == []
        assert store.list_plans() == []

    def test_at_threshold_produces_one_candidate(self):
        miner = TranscriptMiner(min_cluster_size=3)
        store = PlanStore()
        transcripts = [
            _transcript("查北京天气算1+1", "北京", "1+1"),
            _transcript("查上海天气算2+2", "上海", "2+2"),
            _transcript("查广州天气算3+3", "广州", "3+3"),
        ]
        touched = miner.mine(transcripts, store)
        assert len(touched) == 1
        plans = store.list_plans()
        assert len(plans) == 1
        assert [s.tool for s in plans[0].steps] == ["weather", "calc"]
        assert plans[0].steps[0].kwargs_template == {"city": "$city"}
        assert plans[0].steps[1].kwargs_template == {"expression": "$expression"}
        # All three source phrasings became reusable anchors.
        assert len(plans[0].source_queries) == 3

    def test_mined_candidate_starts_unapproved_and_unpromoted(self):
        miner = TranscriptMiner(min_cluster_size=1)
        store = PlanStore(auto_reuse=False)
        touched = miner.mine([_transcript("查北京天气算1+1", "北京", "1+1")], store)
        plan = store.get(touched[0])
        assert plan.approved is False
        assert plan.promoted is False
        assert plan.successes == 0

    def test_different_shapes_cluster_independently(self):
        miner = TranscriptMiner(min_cluster_size=2)
        store = PlanStore()
        transcripts = [
            _transcript("查北京天气算1+1", "北京", "1+1"),
            _transcript("查上海天气算2+2", "上海", "2+2"),
            ReactTranscript(user_input="订一张去伦敦的机票", actions=[
                ("book_flight", {"dest": "伦敦"}),
            ]),  # single action: not a chain, dropped even at cluster size 1
        ]
        touched = miner.mine(transcripts, store)
        assert len(touched) == 1
        assert len(store.list_plans()) == 1

    def test_single_action_transcripts_are_never_mined(self):
        miner = TranscriptMiner(min_cluster_size=1)
        store = PlanStore()
        transcripts = [
            ReactTranscript(user_input="来点天气", actions=[("weather", {"city": "北京"})]),
            ReactTranscript(user_input="来点天气2", actions=[("weather", {"city": "上海"})]),
        ]
        touched = miner.mine(transcripts, store)
        assert touched == []
        assert store.list_plans() == []

    def test_chains_longer_than_max_steps_are_skipped(self):
        miner = TranscriptMiner(min_cluster_size=1, max_steps=2)
        store = PlanStore()
        long_transcript = ReactTranscript(
            user_input="do a lot", actions=[("a", {}), ("b", {}), ("c", {})],
        )
        touched = miner.mine([long_transcript], store)
        assert touched == []
        assert store.list_plans() == []

    def test_repeated_mine_calls_accumulate_anchors_on_same_candidate(self):
        miner = TranscriptMiner(min_cluster_size=1)
        store = PlanStore()
        first = miner.mine([_transcript("查北京天气算1+1", "北京", "1+1")], store)
        second = miner.mine([_transcript("查上海天气算2+2", "上海", "2+2")], store)
        assert first == second  # same shape -> same candidate, not a new one
        assert len(store.list_plans()) == 1
        assert len(store.get(first[0]).source_queries) == 2


# ---------------------------------------------------------------------------
# Agent integration: enable_transcript_mining wiring + mine_scenario_candidates()
# ---------------------------------------------------------------------------

# Each phrasing scripts its own weather(city) -> calc(expr) chain — the
# values a real ReAct loop would have parsed out of that specific request.
_CHAIN_VALUES = {
    "查北京天气算1+1": ("北京", "1+1"),
    "查上海天气算2+2": ("上海", "2+2"),
    "查广州天气算3+3": ("广州", "3+3"),
}


class _WeatherTool(Tool):
    def __init__(self):
        super().__init__(name="weather", description="query city weather")

    async def execute(self, context, **kwargs) -> ToolOutput:
        return ToolOutput(success=True, result="sunny")


class _CalcTool(Tool):
    def __init__(self):
        super().__init__(name="calc", description="evaluate arithmetic")

    async def execute(self, context, **kwargs) -> ToolOutput:
        return ToolOutput(success=True, result="2")


class _ForcedReactAgent(Agent):
    """Always classifies REACT and scripts a deterministic 2-tool chain,
    keyed off the request text, without depending on model call ordering."""

    async def _classify_intent(self, user_input, context):  # type: ignore[override]
        return IntentResult(level=IntentLevel.REACT, confidence=1.0)

    async def _generate_thought(self, context, model, accumulated_context):  # type: ignore[override]
        city, expr = _CHAIN_VALUES[context.user_input]
        if context.current_iteration == 1:
            return f'Thought: need weather.\nAction: weather\nAction Input: {{"city": "{city}"}}'
        if context.current_iteration == 2:
            return f'Thought: now calc.\nAction: calc\nAction Input: {{"expression": "{expr}"}}'
        return "Final Answer: done"


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
    return agent


class TestAgentTranscriptMining:
    @pytest.mark.asyncio
    async def test_disabled_by_default_no_transcripts_logged(self):
        agent = _make_react_agent()
        await agent.run("查北京天气算1+1")
        assert agent._react_transcripts == []
        assert agent.mine_scenario_candidates() == []

    @pytest.mark.asyncio
    async def test_enabled_logs_and_mines_repeated_chain(self):
        agent = _make_react_agent(
            enable_transcript_mining=True, mining_min_cluster_size=2,
        )

        await agent.run("查北京天气算1+1")
        assert len(agent._react_transcripts) == 1

        touched = agent.mine_scenario_candidates()
        assert touched == []  # only one occurrence so far — below cluster size
        assert agent._react_transcripts == []  # batch drained regardless

        await agent.run("查上海天气算2+2")
        await agent.run("查广州天气算3+3")
        touched = agent.mine_scenario_candidates()
        assert len(touched) == 1
        candidates = agent.list_plan_candidates()
        assert len(candidates) == 1
        assert [s["tool"] for s in candidates[0]["steps"]] == ["weather", "calc"]
        assert candidates[0]["promoted"] is False
