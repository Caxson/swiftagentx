"""
Benchmark scenario definitions.

Each scenario builds a fresh Agent, runs N iterations, and returns timing,
LLM-call, and success statistics. Scenarios are written to exercise specific
execution paths through the framework.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from swiftagentx import (
    Agent,
    Document,
    MemoryKnowledgeBase,
    ModelTier,
    ScenarioConfig,
    SwiftAgentConfig,
    Tool,
    ToolChainStep,
    ToolOutput,
)

from .mock_provider import MockCallStats, MockModelClient


@dataclass
class ScenarioResult:
    name: str
    iterations: int
    latencies_ms: list[float] = field(default_factory=list)
    llm_calls_per_iter: list[int] = field(default_factory=list)
    tokens_per_iter: list[int] = field(default_factory=list)
    failures: int = 0
    cache_hits: int = 0
    notes: str = ""

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        k = (len(sorted_lat) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_lat) - 1)
        if f == c:
            return sorted_lat[f]
        return sorted_lat[f] + (sorted_lat[c] - sorted_lat[f]) * (k - f)

    @property
    def p50_ms(self) -> float:
        return self.percentile(50)

    @property
    def p95_ms(self) -> float:
        return self.percentile(95)

    @property
    def p99_ms(self) -> float:
        return self.percentile(99)

    @property
    def avg_llm_calls(self) -> float:
        return sum(self.llm_calls_per_iter) / len(self.llm_calls_per_iter) if self.llm_calls_per_iter else 0.0

    @property
    def avg_tokens(self) -> float:
        return sum(self.tokens_per_iter) / len(self.tokens_per_iter) if self.tokens_per_iter else 0.0

    @property
    def success_rate(self) -> float:
        total = self.iterations
        return (total - self.failures) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "avg_llm_calls": round(self.avg_llm_calls, 2),
            "avg_tokens": round(self.avg_tokens, 2),
            "success_rate": round(self.success_rate, 4),
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Helper tools for scenarios
# ---------------------------------------------------------------------------


class WeatherTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="weather", description="Get the current weather for a city.")

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        city = kwargs.get("query") or kwargs.get("city") or "unknown"
        return ToolOutput(success=True, result=f"{city}: sunny, 22°C")


class FlakyTool(Tool):
    """Fails the first time then succeeds — exercises ReAct recovery."""

    def __init__(self) -> None:
        super().__init__(name="flaky", description="Fails once before succeeding.")
        self._called = 0

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        self._called += 1
        if self._called == 1:
            return ToolOutput(success=False, result=None, error="transient failure")
        return ToolOutput(success=True, result="ok")


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _build_agent(
    light_stats: MockCallStats,
    heavy_stats: MockCallStats,
    *,
    light_response_map: dict[str, str] | None = None,
    heavy_response_map: dict[str, str] | None = None,
    register_weather_scenario: bool = False,
) -> Agent:
    light = MockModelClient(tier="light", stats=light_stats)
    heavy = MockModelClient(tier="heavy", stats=heavy_stats)
    if light_response_map:
        light.set_response_map(light_response_map)
    if heavy_response_map:
        heavy.set_response_map(heavy_response_map)

    agent = Agent(
        models={ModelTier.LIGHT: light, ModelTier.HEAVY: heavy},
        config=SwiftAgentConfig(name="bench-agent", max_iterations=4, enable_cache=True),
    )
    agent.register_tool(WeatherTool())
    if register_weather_scenario:
        agent.register_scenario(
            "weather",
            ScenarioConfig(
                name="Weather",
                description="weather lookup",
                triggers=["weather", "temperature"],
                tool_chain=[ToolChainStep(tool="weather", query_template="$city")],
                cache_ttl=300,
                output_type="direct",
            ),
        )
    return agent


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------


async def _run_iterations(
    name: str,
    iterations: int,
    fn: Callable[[int], Awaitable[bool]],
    light_stats: MockCallStats,
    heavy_stats: MockCallStats,
    notes: str = "",
) -> ScenarioResult:
    result = ScenarioResult(name=name, iterations=iterations, notes=notes)
    for i in range(iterations):
        light_stats.reset()
        heavy_stats.reset()
        t0 = time.perf_counter()
        try:
            ok = await fn(i)
        except Exception:
            ok = False
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result.latencies_ms.append(elapsed_ms)
        result.llm_calls_per_iter.append(light_stats.total_calls + heavy_stats.total_calls)
        result.tokens_per_iter.append(light_stats.total_tokens + heavy_stats.total_tokens)
        if not ok:
            result.failures += 1
    return result


async def scenario_simple_qa(iterations: int = 20) -> ScenarioResult:
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(
        light_stats, heavy_stats,
        light_response_map={"classify": "level=3"},
        heavy_response_map={"capital": "Paris is the capital of France."},
    )

    async def one(i: int) -> bool:
        resp = await agent.run(
            "What is the capital of France?", user_id=f"u{i}", session_id=f"s{i}",
        )
        return bool(resp.answer)

    return await _run_iterations(
        "simple_qa", iterations, one, light_stats, heavy_stats,
        notes="Direct LLM response — 1 LIGHT classify + 1 HEAVY answer expected",
    )


async def scenario_kb_exact_match(iterations: int = 20) -> ScenarioResult:
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(light_stats, heavy_stats)
    kb = MemoryKnowledgeBase()
    await kb.add_documents([
        Document(doc_id="faq-1", content="Returns policy: 7-day no-questions-asked returns."),
        Document(doc_id="faq-2", content="Points can be redeemed in the member store."),
    ])
    agent.set_knowledge_base(kb)

    async def one(i: int) -> bool:
        resp = await agent.run(
            "Returns policy: 7-day no-questions-asked returns.",
            user_id=f"u{i}", session_id=f"s{i}",
        )
        return bool(resp.answer)

    return await _run_iterations(
        "kb_exact_match", iterations, one, light_stats, heavy_stats,
        notes="Exact KB match — should short-circuit to ~0 LLM calls",
    )


async def scenario_single_tool_call(iterations: int = 20) -> ScenarioResult:
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(
        light_stats, heavy_stats, register_weather_scenario=True,
        light_response_map={"classify": "level=2 scenario=weather"},
    )

    async def one(i: int) -> bool:
        resp = await agent.run(
            "What is the weather in Beijing?", user_id=f"u{i}", session_id=f"s{i}",
            city="Beijing",
        )
        return bool(resp.answer)

    return await _run_iterations(
        "single_tool_call", iterations, one, light_stats, heavy_stats,
        notes="Scenario engine — 1 LIGHT classify + 1 tool exec, no ReAct",
    )


async def scenario_multi_turn_dialog(iterations: int = 10) -> ScenarioResult:
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(
        light_stats, heavy_stats,
        light_response_map={"classify": "level=3"},
        heavy_response_map={"continue": "Let me continue the previous topic."},
    )

    async def one(i: int) -> bool:
        session = f"s{i}"
        user = f"u{i}"
        turns = ["Hi there.", "Tell me about yourself.", "Continue.", "What else?", "Thanks."]
        for turn in turns:
            await agent.run(turn, user_id=user, session_id=session)
        return True

    return await _run_iterations(
        "multi_turn_dialog", iterations, one, light_stats, heavy_stats,
        notes="5 sequential turns sharing session memory",
    )


async def scenario_rag_query(iterations: int = 15) -> ScenarioResult:
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(
        light_stats, heavy_stats,
        light_response_map={"classify": "level=3"},
        heavy_response_map={"returns": "Returns are accepted within 7 days."},
    )
    kb = MemoryKnowledgeBase()
    await kb.add_documents([
        Document(doc_id="r1", content="Returns are accepted within 7 days of purchase."),
        Document(doc_id="r2", content="Refunds are processed within 5 business days."),
    ])
    agent.set_knowledge_base(kb)

    async def one(i: int) -> bool:
        resp = await agent.run(
            "What is your returns policy?", user_id=f"u{i}", session_id=f"s{i}",
        )
        return bool(resp.answer)

    return await _run_iterations(
        "rag_query", iterations, one, light_stats, heavy_stats,
        notes="KB search → LLM synthesis",
    )


async def scenario_concurrency_qps(iterations: int = 1) -> ScenarioResult:
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(light_stats, heavy_stats)

    concurrency = 50

    async def one(i: int) -> bool:
        async def single(j: int) -> bool:
            resp = await agent.run(
                f"ping {j}", user_id=f"u{j}", session_id=f"s{j}",
            )
            return bool(resp.answer)

        results = await asyncio.gather(*(single(j) for j in range(concurrency)), return_exceptions=True)
        ok_count = sum(1 for r in results if r is True)
        return ok_count >= concurrency * 0.95

    res = await _run_iterations(
        "concurrency_qps", iterations, one, light_stats, heavy_stats,
        notes=f"{concurrency} concurrent requests; latency is wall-clock total",
    )
    return res


async def scenario_cache_repeated(iterations: int = 1) -> ScenarioResult:
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(light_stats, heavy_stats)
    kb = MemoryKnowledgeBase()
    await kb.add_documents([Document(doc_id="d", content="cached answer content")])
    agent.set_knowledge_base(kb)

    async def one(i: int) -> bool:
        for _ in range(100):
            await agent.run(
                "cached answer content", user_id="ucache", session_id="scache",
            )
        return True

    return await _run_iterations(
        "cache_repeated", iterations, one, light_stats, heavy_stats,
        notes="100 identical KB-exact queries; LLM calls should stay near 0",
    )


async def scenario_multi_tool_chain(iterations: int = 15) -> ScenarioResult:
    """Two-step scenario chain: weather → format result."""
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(
        light_stats, heavy_stats,
        light_response_map={"classify": "level=2 scenario=weather_then_format"},
    )

    agent.register_scenario(
        "weather_then_format",
        ScenarioConfig(
            name="WeatherThenFormat",
            description="lookup then format",
            triggers=["weather", "temperature", "forecast"],
            tool_chain=[
                ToolChainStep(tool="weather", query_template="$city", extract_to="raw"),
                ToolChainStep(tool="weather", query_template="$raw", extract_to="formatted"),
            ],
            cache_ttl=300,
            output_type="llm_processed",
        ),
    )

    async def one(i: int) -> bool:
        resp = await agent.run(
            "Get the weather forecast for Beijing.", user_id=f"u{i}", session_id=f"s{i}",
        )
        return bool(resp.answer)

    return await _run_iterations(
        "multi_tool_chain", iterations, one, light_stats, heavy_stats,
        notes="2-step scenario tool chain, no ReAct loop",
    )


async def scenario_complex_react(iterations: int = 10) -> ScenarioResult:
    """Forces full ReAct loop by classifying as 'react' intent."""
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(
        light_stats,
        heavy_stats,
        light_response_map={"classify": "level=1"},
        heavy_response_map={
            "think": "Thought: I should call weather. Action: weather. Action Input: {\"query\":\"Beijing\"}",
            "observation": "Final Answer: Beijing is sunny.",
        },
    )
    agent.config.max_iterations = 3

    async def one(i: int) -> bool:
        resp = await agent.run(
            "Reason step by step about Beijing weather, then answer.",
            user_id=f"u{i}", session_id=f"s{i}",
        )
        return bool(resp.answer)

    return await _run_iterations(
        "complex_react", iterations, one, light_stats, heavy_stats,
        notes="Full ReAct loop, ~3 heavy-model iterations",
    )


async def scenario_error_recovery(iterations: int = 10) -> ScenarioResult:
    light_stats, heavy_stats = MockCallStats(), MockCallStats()
    agent = _build_agent(
        light_stats, heavy_stats,
        light_response_map={"classify": "level=1"},
        heavy_response_map={"flaky": "Final Answer: Eventually recovered."},
    )
    agent.register_tool(FlakyTool())

    async def one(i: int) -> bool:
        resp = await agent.run(
            "Run the flaky tool.", user_id=f"u{i}", session_id=f"s{i}",
        )
        return bool(resp.answer)

    return await _run_iterations(
        "error_recovery", iterations, one, light_stats, heavy_stats,
        notes="One tool returns failure once; ReAct should recover",
    )


ALL_SCENARIOS: list[Callable[[], Awaitable[ScenarioResult]]] = [
    scenario_simple_qa,
    scenario_kb_exact_match,
    scenario_single_tool_call,
    scenario_multi_tool_chain,
    scenario_multi_turn_dialog,
    scenario_rag_query,
    scenario_complex_react,
    scenario_concurrency_qps,
    scenario_cache_repeated,
    scenario_error_recovery,
]
