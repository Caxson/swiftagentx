# SwiftAgentX Benchmark Suite

Reproducible latency and LLM-call benchmarks for the SwiftAgentX framework.
Measures the impact of the tiered execution architecture (cache → scenario →
ReAct → direct) across ten representative request patterns.

## Why these benchmarks exist

The README claims "sub-second response for 80% of common requests" and "fewer
LLM calls than a vanilla ReAct loop." This suite is what backs those claims
with reproducible numbers. If you want to challenge them, run the benchmarks
yourself.

## Quick start

```bash
pip install -e ".[dev,benchmark]"

# Mock LLM mode — deterministic, no API key required, runs in seconds
python benchmarks/run_benchmarks.py --mode mock

# Real OpenAI-compatible mode — needs LLM_API_KEY env var
LLM_API_KEY=... LLM_BASE_URL=https://api.openai.com/v1 LLM_MODEL=gpt-4o-mini \
    python benchmarks/run_benchmarks.py --mode real
```

Results land in `benchmark_results.json` and a PNG chart per scenario.

## What the suite measures

| Metric | Why it matters |
|--------|----------------|
| P50 / P95 / P99 latency | Tail latency is what users feel, not the mean |
| LLM calls per request | Every call is dollars and seconds |
| Token usage | The other half of "every call is dollars" |
| Cache hit rate | Validates the three-level cache claim |
| Success rate | A fast wrong answer is worse than a slow right one |

## The ten scenarios

| # | Scenario | What it tests |
|---|----------|---------------|
| 1 | `simple_qa` | Direct LLM response path (no tools, no cache) |
| 2 | `kb_exact_match` | KB short-circuit — should be ~0 ms, 0 LLM calls |
| 3 | `single_tool_call` | One tool invocation through scenario engine |
| 4 | `multi_tool_chain` | Chained tools, scenario engine |
| 5 | `multi_turn_dialog` | Session memory across 5 turns |
| 6 | `rag_query` | KB search → answer (LIGHT classify + HEAVY synth) |
| 7 | `complex_react` | Forces full ReAct loop, ≥3 iterations |
| 8 | `concurrency_qps` | 50 parallel requests, measures throughput degradation |
| 9 | `cache_repeated` | Same query 100x, measures L1 cache amortization |
| 10| `error_recovery` | Tool fails → ReAct should recover |

Each scenario specifies `expected_max_p95_ms` and `expected_llm_calls`. The
runner fails the suite if reality exceeds the bound by more than 50%.

## Comparing against LangChain

`benchmarks/compare_langchain.py` runs the same scenarios against a minimal
LangChain agent (when `langchain` is installed). The comparison chart is
written to `benchmark_comparison.png` and embedded in the README.

We don't pretend LangChain is a bad framework — it's the standard. We're
demonstrating that domain-specific architectural choices (tiered execution,
scenario routing) win on latency for production patterns where 80% of
requests are predictable.

## Reproducibility

Mock mode is fully deterministic: same seed, same numbers across runs.
Real mode varies with provider latency — we report the median of 5 runs.

All raw measurements stay in `benchmark_results.json` so anyone can re-plot
or re-aggregate.
