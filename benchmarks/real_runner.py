"""
Real-LLM benchmark runner using DashScope Qwen models.

Drives the same execution paths as the mock runner but talks to a real
OpenAI-compatible endpoint (DashScope is the default). Iteration counts
are kept low because each call costs real money; the design point is
"enough samples to compute meaningful P50/P95, but not 100".

Env vars consumed:

  DASHSCOPE_API_KEY    required
  DASHSCOPE_BASE_URL   default https://dashscope.aliyuncs.com/compatible-mode/v1
  LIGHT_MODEL          default qwen-flash    (cheap, fast — for classification)
  HEAVY_MODEL          default qwen-turbo    (slightly bigger — for answers)

Run::

    python benchmarks/real_runner.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swiftagentx import (  # noqa: E402
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
from swiftagentx.providers.openai_compatible import OpenAICompatibleProvider  # noqa: E402

from benchmarks.scenarios import ScenarioResult, _run_iterations  # noqa: E402


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------


def _build_provider(model_name: str) -> OpenAICompatibleProvider:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY env var is not set.", file=sys.stderr)
        sys.exit(2)
    base = os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return OpenAICompatibleProvider(api_key=api_key, model=model_name, api_base=base)


# ---------------------------------------------------------------------------
# Telemetry — count actual LLM calls via a wrapping proxy
# ---------------------------------------------------------------------------


class _CountingProvider:
    """Wraps an OpenAICompatibleProvider, counts chat() invocations + tokens."""

    def __init__(self, inner: OpenAICompatibleProvider) -> None:
        self.inner = inner
        # Mirror the attributes the framework reads off ModelClient.
        self.api_key = inner.api_key
        self.model = inner.model
        self.temperature = inner.temperature
        self.max_tokens = inner.max_tokens
        self.timeout_seconds = inner.timeout_seconds
        self.calls = 0
        self.total_tokens = 0

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        response = await self.inner.chat(messages, **kw)
        self.calls += 1
        self.total_tokens += getattr(response, "tokens_used", 0)
        return response

    async def complete(self, prompt: str, **kw: Any) -> Any:
        response = await self.inner.complete(prompt, **kw)
        self.calls += 1
        self.total_tokens += getattr(response, "tokens_used", 0)
        return response

    async def stream_chat(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        async for chunk in self.inner.stream_chat(messages, **kw):
            yield chunk

    async def stream_complete(self, prompt: str, **kw: Any) -> Any:
        async for chunk in self.inner.stream_complete(prompt, **kw):
            yield chunk

    def reset(self) -> None:
        self.calls = 0
        self.total_tokens = 0


# ---------------------------------------------------------------------------
# Helper tools
# ---------------------------------------------------------------------------


class _WeatherTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="weather", description="Get the current weather for a city.")

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        city = kwargs.get("query") or kwargs.get("city") or "Beijing"
        return ToolOutput(success=True, result=f"{city}: sunny, 22 C")


class _FlakyTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="flaky", description="Fails once before succeeding.")
        self._n = 0

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        self._n += 1
        if self._n == 1:
            return ToolOutput(success=False, result=None, error="transient failure")
        return ToolOutput(success=True, result="ok")


# ---------------------------------------------------------------------------
# Agent factories — share a single counting provider across scenarios so we
# can sum LLM calls accurately per iteration.
# ---------------------------------------------------------------------------


def _build_agent(
    light: _CountingProvider,
    heavy: _CountingProvider,
    *,
    register_weather_scenario: bool = False,
    extra_scenarios: list[tuple[str, ScenarioConfig]] | None = None,
    extra_tools: list[Tool] | None = None,
) -> Agent:
    agent = Agent(
        models={ModelTier.LIGHT: light, ModelTier.HEAVY: heavy},  # type: ignore[arg-type]
        config=SwiftAgentConfig(
            name="real-bench-agent",
            max_iterations=4,
            enable_cache=True,
            memory_enable_topic_change_hook=False,  # avoid extra LIGHT calls
        ),
    )
    agent.register_tool(_WeatherTool())
    if extra_tools:
        for t in extra_tools:
            agent.register_tool(t)
    if register_weather_scenario:
        agent.register_scenario(
            "weather",
            ScenarioConfig(
                name="Weather", description="weather lookup",
                triggers=["weather", "temperature", "forecast"],
                tool_chain=[ToolChainStep(tool="weather", query_template="$city")],
                cache_ttl=300, output_type="direct",
            ),
        )
    if extra_scenarios:
        for sid, sc in extra_scenarios:
            agent.register_scenario(sid, sc)
    return agent


# ---------------------------------------------------------------------------
# Real scenarios — each returns ScenarioResult with real latency/call data.
# ---------------------------------------------------------------------------


async def real_simple_qa(light: _CountingProvider, heavy: _CountingProvider,
                         iterations: int) -> ScenarioResult:
    agent = _build_agent(light, heavy)
    result = ScenarioResult(name="simple_qa", iterations=iterations,
                            notes="DIRECT path: 1 LIGHT classify + 1 HEAVY answer")
    for i in range(iterations):
        light.reset()
        heavy.reset()
        t0 = time.perf_counter()
        resp = await agent.run(
            "What is the capital of France?", user_id=f"u{i}", session_id=f"sqa{i}",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        result.latencies_ms.append(elapsed)
        result.llm_calls_per_iter.append(light.calls + heavy.calls)
        result.tokens_per_iter.append(light.total_tokens + heavy.total_tokens)
        if not resp.answer:
            result.failures += 1
    return result


async def real_kb_exact_match(light: _CountingProvider, heavy: _CountingProvider,
                              iterations: int) -> ScenarioResult:
    agent = _build_agent(light, heavy)
    kb = MemoryKnowledgeBase()
    await kb.add_documents([
        Document(doc_id="faq-1", content="Returns policy: 7-day no-questions-asked returns."),
        Document(doc_id="faq-2", content="Points can be redeemed in the member store."),
    ])
    agent.set_knowledge_base(kb)

    result = ScenarioResult(name="kb_exact_match", iterations=iterations,
                            notes="Pipeline short-circuit; ~0 LLM calls")
    for i in range(iterations):
        light.reset()
        heavy.reset()
        t0 = time.perf_counter()
        resp = await agent.run(
            "Returns policy: 7-day no-questions-asked returns.",
            user_id=f"u{i}", session_id=f"kb{i}",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        result.latencies_ms.append(elapsed)
        result.llm_calls_per_iter.append(light.calls + heavy.calls)
        result.tokens_per_iter.append(light.total_tokens + heavy.total_tokens)
        if not resp.answer:
            result.failures += 1
    return result


async def real_scenario_shortcut(light: _CountingProvider, heavy: _CountingProvider,
                                 iterations: int) -> ScenarioResult:
    agent = _build_agent(light, heavy, register_weather_scenario=True)
    result = ScenarioResult(name="scenario_shortcut", iterations=iterations,
                            notes="Scenario tool chain; 1 LIGHT call (no HEAVY)")
    for i in range(iterations):
        light.reset()
        heavy.reset()
        t0 = time.perf_counter()
        resp = await agent.run(
            "What is the weather in Beijing?",
            user_id=f"u{i}", session_id=f"sc{i}", city="Beijing",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        result.latencies_ms.append(elapsed)
        result.llm_calls_per_iter.append(light.calls + heavy.calls)
        result.tokens_per_iter.append(light.total_tokens + heavy.total_tokens)
        if not resp.answer:
            result.failures += 1
    return result


async def real_cache_hit(light: _CountingProvider, heavy: _CountingProvider,
                         iterations: int) -> ScenarioResult:
    """Repeated KB-exact query: first hit pays, subsequent hits are free."""
    agent = _build_agent(light, heavy)
    kb = MemoryKnowledgeBase()
    await kb.add_documents([Document(doc_id="d", content="cached answer content")])
    agent.set_knowledge_base(kb)

    # Warm up.
    await agent.run("cached answer content", user_id="warm", session_id="warm")

    result = ScenarioResult(name="cache_hit", iterations=iterations,
                            notes="100 repeated KB-exact queries — warm pipeline short-circuit")
    for i in range(iterations):
        light.reset()
        heavy.reset()
        t0 = time.perf_counter()
        resp = await agent.run("cached answer content",
                               user_id="warm", session_id="warm")
        elapsed = (time.perf_counter() - t0) * 1000
        result.latencies_ms.append(elapsed)
        result.llm_calls_per_iter.append(light.calls + heavy.calls)
        result.tokens_per_iter.append(light.total_tokens + heavy.total_tokens)
        if not resp.answer:
            result.failures += 1
    return result


async def real_react_complex(light: _CountingProvider, heavy: _CountingProvider,
                             iterations: int) -> ScenarioResult:
    """Force a full ReAct loop — the most expensive path."""
    agent = _build_agent(light, heavy)
    result = ScenarioResult(
        name="react_complex", iterations=iterations,
        notes="Full ReAct loop — multi-step reasoning (the expensive path)",
    )
    for i in range(iterations):
        light.reset()
        heavy.reset()
        t0 = time.perf_counter()
        resp = await agent.run(
            "Think step by step, then explain why the sky is blue in 2 sentences.",
            user_id=f"u{i}", session_id=f"rc{i}",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        result.latencies_ms.append(elapsed)
        result.llm_calls_per_iter.append(light.calls + heavy.calls)
        result.tokens_per_iter.append(light.total_tokens + heavy.total_tokens)
        if not resp.answer:
            result.failures += 1
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


SCENARIOS: list[
    Callable[[_CountingProvider, _CountingProvider, int], Awaitable[ScenarioResult]]
] = [
    real_kb_exact_match,
    real_scenario_shortcut,
    real_cache_hit,
    real_simple_qa,
    real_react_complex,
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5,
                        help="Iterations per scenario (default 5 — keeps cost low)")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--chart", type=Path, default=Path("benchmark_comparison.png"))
    parser.add_argument("--light-model", default=os.environ.get("LIGHT_MODEL", "qwen-flash"))
    parser.add_argument("--heavy-model", default=os.environ.get("HEAVY_MODEL", "qwen-turbo"))
    parser.add_argument("--no-chart", action="store_true")
    args = parser.parse_args()

    light = _CountingProvider(_build_provider(args.light_model))
    heavy = _CountingProvider(_build_provider(args.heavy_model))

    print(f"Real-LLM benchmark — LIGHT={args.light_model}, HEAVY={args.heavy_model},"
          f" iterations={args.iterations}")

    results: list[ScenarioResult] = []
    for fn in SCENARIOS:
        print(f"  running {fn.__name__} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            r = await fn(light, heavy, args.iterations)
        except Exception as exc:
            print(f"FAILED ({type(exc).__name__}: {exc})")
            continue
        dt = time.perf_counter() - t0
        results.append(r)
        print(f"done in {dt:.1f}s "
              f"(P50={r.p50_ms:.0f}ms, calls={r.avg_llm_calls:.1f},"
              f" tokens={r.avg_tokens:.0f})")

    # Tabular summary
    print("\n" + "-" * 92)
    print(f"{'scenario':<22} {'P50':>8} {'P95':>8} {'P99':>8}"
          f" {'LLM calls':>10} {'tokens':>10} {'success':>10}")
    print("-" * 92)
    for r in results:
        print(f"{r.name:<22} {r.p50_ms:>7.0f}ms {r.p95_ms:>7.0f}ms"
              f" {r.p99_ms:>7.0f}ms {r.avg_llm_calls:>10.1f}"
              f" {r.avg_tokens:>10.0f} {r.success_rate*100:>9.0f}%")
    print("-" * 92)
    print(f"\nTotal LIGHT calls: {light.calls}   tokens: {light.total_tokens}")
    print(f"Total HEAVY calls: {heavy.calls}   tokens: {heavy.total_tokens}")

    # JSON report
    args.output.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": {"light": args.light_model, "heavy": args.heavy_model},
        "iterations_per_scenario": args.iterations,
        "scenarios": [r.to_dict() for r in results],
        "totals": {
            "light_calls": light.calls, "light_tokens": light.total_tokens,
            "heavy_calls": heavy.calls, "heavy_tokens": heavy.total_tokens,
        },
    }, indent=2))
    print(f"\nWrote {args.output}")

    if not args.no_chart:
        try:
            _emit_chart(results, args.chart)
        except Exception as exc:  # noqa: BLE001
            print(f"chart skipped: {exc}", file=sys.stderr)

    return 0


def _emit_chart(results: list[ScenarioResult], path: Path) -> None:
    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt

    names = [r.name for r in results]
    p50 = [r.p50_ms for r in results]
    p95 = [r.p95_ms for r in results]
    calls = [r.avg_llm_calls for r in results]

    fig, (ax_lat, ax_calls) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1]},
    )

    x = list(range(len(names)))
    width = 0.35

    def _fmt_latency(ms: float) -> str:
        if ms >= 1000:
            return f"{ms / 1000:.1f}s"
        if ms >= 1:
            return f"{ms:.0f}ms"
        return f"{ms:.2f}ms"  # sub-millisecond tiers (cache / KB hit)

    bars_p50 = ax_lat.bar([i - width/2 for i in x], p50, width=width, label="P50", color="#0ea5e9")
    bars_p95 = ax_lat.bar([i + width/2 for i in x], p95, width=width, label="P95", color="#f59e0b")
    ax_lat.set_ylabel("Latency (ms, log scale)")
    ax_lat.set_yscale("log")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(names, rotation=15, ha="right")
    ax_lat.set_title("SwiftAgentX real-LLM benchmark (DashScope Qwen)")
    ax_lat.legend()
    ax_lat.grid(True, axis="y", alpha=0.3)
    # Value labels on every bar — without them the two sub-millisecond tiers
    # (kb_exact_match, cache_hit) are squashed to the axis floor on the log
    # scale and unreadable, which buries the framework's strongest result.
    for bars, vals in ((bars_p50, p50), (bars_p95, p95)):
        for rect, v in zip(bars, vals):
            ax_lat.annotate(_fmt_latency(v),
                            (rect.get_x() + rect.get_width() / 2, max(v, 1e-3)),
                            ha="center", va="bottom", fontsize=7,
                            xytext=(0, 2), textcoords="offset points")

    ax_calls.bar(x, calls, color="#d97706")
    ax_calls.set_ylabel("Avg LLM calls / request")
    ax_calls.set_xticks(x)
    ax_calls.set_xticklabels(names, rotation=15, ha="right")
    ax_calls.set_title("LLM calls per request — the cost story")
    ax_calls.grid(True, axis="y", alpha=0.3)

    for i, n in enumerate(calls):
        ax_calls.text(i, n + 0.05, f"{n:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
