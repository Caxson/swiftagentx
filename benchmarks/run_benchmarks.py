#!/usr/bin/env python
"""
Run the SwiftAgentX benchmark suite.

Usage
-----

  python benchmarks/run_benchmarks.py --mode mock
  python benchmarks/run_benchmarks.py --mode real

Mock mode uses a deterministic latency-simulating fake LLM and produces
the same numbers across runs (good for CI). Real mode talks to an
OpenAI-compatible endpoint via env vars:

  LLM_API_KEY     — required
  LLM_BASE_URL    — default https://api.openai.com/v1
  LLM_MODEL_LIGHT — default gpt-4o-mini
  LLM_MODEL_HEAVY — default gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure both the package and the benchmarks/ directory are importable
# when run from a source checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.scenarios import ALL_SCENARIOS, ScenarioResult  # noqa: E402


def _print_table(results: list[ScenarioResult]) -> None:
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    rows = []
    for r in results:
        rows.append([
            r.name,
            r.iterations,
            f"{r.p50_ms:.0f}",
            f"{r.p95_ms:.0f}",
            f"{r.p99_ms:.0f}",
            f"{r.avg_llm_calls:.1f}",
            f"{r.avg_tokens:.0f}",
            f"{r.success_rate * 100:.0f}%",
        ])

    headers = ["scenario", "n", "p50 ms", "p95 ms", "p99 ms", "LLM calls", "tokens", "success"]
    if tabulate:
        print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
    else:
        # ASCII fallback when tabulate isn't installed
        print("\n" + " | ".join(headers))
        print("-" * 90)
        for row in rows:
            print(" | ".join(str(c) for c in row))


def _save_json(results: list[ScenarioResult], path: Path) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenarios": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {path}")


def _maybe_plot(results: list[ScenarioResult], out_dir: Path) -> None:
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart. pip install swiftagentx[benchmark]")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    names = [r.name for r in results]
    p50 = [r.p50_ms for r in results]
    p95 = [r.p95_ms for r in results]
    p99 = [r.p99_ms for r in results]
    calls = [r.avg_llm_calls for r in results]

    fig, (ax_lat, ax_calls) = plt.subplots(
        2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]}
    )

    x = range(len(names))
    width = 0.27
    ax_lat.bar([i - width for i in x], p50, width=width, label="P50")
    ax_lat.bar(list(x), p95, width=width, label="P95")
    ax_lat.bar([i + width for i in x], p99, width=width, label="P99")
    ax_lat.set_ylabel("Latency (ms, log scale)")
    ax_lat.set_yscale("log")
    ax_lat.set_xticks(list(x))
    ax_lat.set_xticklabels(names, rotation=30, ha="right")
    ax_lat.set_title("SwiftAgentX latency by scenario")
    ax_lat.legend()
    ax_lat.grid(True, axis="y", alpha=0.3)

    ax_calls.bar(list(x), calls, color="#d97706")
    ax_calls.set_ylabel("Avg LLM calls / request")
    ax_calls.set_xticks(list(x))
    ax_calls.set_xticklabels(names, rotation=30, ha="right")
    ax_calls.set_title("LLM calls per request")
    ax_calls.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "benchmark_latency.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")


async def _run_real_mode(args: argparse.Namespace) -> list[ScenarioResult]:
    """Run benchmarks against a real OpenAI-compatible endpoint."""
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("ERROR: real mode needs LLM_API_KEY env var", file=sys.stderr)
        sys.exit(2)

    # Real-mode runner is intentionally not wired into ALL_SCENARIOS yet —
    # it would burn real $ on every CI run and obscure determinism issues.
    # For now, real mode reuses the mock scenarios but you can swap the
    # MockModelClient with OpenAICompatibleProvider in your local fork.
    print("WARNING: real mode currently runs the mock scenarios with mock-LLM "
          "latency. A future patch will route them through OpenAICompatibleProvider.",
          file=sys.stderr)
    return await _run_mock_mode(args)


async def _run_mock_mode(args: argparse.Namespace) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for scenario_fn in ALL_SCENARIOS:
        print(f"  running {scenario_fn.__name__} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            r = await scenario_fn()
        except Exception as exc:
            print(f"FAILED ({type(exc).__name__}: {exc})")
            continue
        dt = time.perf_counter() - t0
        results.append(r)
        print(f"done in {dt:.2f}s (P50={r.p50_ms:.0f}ms, calls={r.avg_llm_calls:.1f})")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["mock", "real"], default="mock",
                   help="mock = deterministic local; real = OpenAI-compatible")
    p.add_argument("--output", type=Path, default=Path("benchmark_results.json"),
                   help="Where to write the JSON report")
    p.add_argument("--chart-dir", type=Path, default=Path("."),
                   help="Where to write PNG charts")
    p.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    print(f"SwiftAgentX benchmark — mode={args.mode}")
    runner = _run_real_mode if args.mode == "real" else _run_mock_mode
    results = await runner(args)
    if not results:
        print("No scenarios completed.", file=sys.stderr)
        return 1
    _print_table(results)
    _save_json(results, args.output)
    if not args.no_chart:
        _maybe_plot(results, args.chart_dir)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
