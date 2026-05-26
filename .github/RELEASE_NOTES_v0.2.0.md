# SwiftAgentX v0.2.0 — Production-readiness pass

> **Draft.** Review and edit before publishing on GitHub Releases / PyPI.
> Do not commit this file (it's a draft for the maintainer).

This release does **not** add new framework features. It hardens what was
already there so the project can stand up to a closer look — from
prospective users, contributors, and yes, interviewers.

## TL;DR

- **Benchmark suite** with ten reproducible scenarios; mock and real LLM modes
- **README rewrite** with a sharper positioning and a head-to-head comparison
  against LangChain / AutoGen / CrewAI
- **GitHub Actions CI** running on Python 3.10 / 3.11 / 3.12 / 3.13
- **CHANGELOG**, **CONTRIBUTING**, **CODE_OF_CONDUCT** added
- Six self-contained `examples/cookbook/` recipes
- A handful of latent bugs fixed (version drift, broken project URLs,
  swallowed exceptions, Python 3.11-only API used under a 3.9 declaration)

The execution architecture — tiered cache → scenario → ReAct → direct, with
LIGHT/HEAVY model routing — is unchanged.

## What's new

### Benchmarks

`benchmarks/run_benchmarks.py` measures ten scenarios and produces a JSON
report plus latency / LLM-call charts. Mock mode is deterministic and runs
in CI; real mode talks to any OpenAI-compatible endpoint.

```bash
pip install -e ".[dev,benchmark]"
python benchmarks/run_benchmarks.py --mode mock
```

The scenarios exercise cache hits, KB exact match, scenario tool chains,
RAG, multi-turn dialog, concurrency (50 parallel requests), repeated
queries (100x), full ReAct loops, and tool-failure recovery.

### Cookbook

`examples/cookbook/` now contains:

- `customer_service_agent.py` — KB + scenario + tool fallback
- `rag_chatbot.py` — multi-turn RAG with a pipeline stage
- `tool_calling_workflow.py` — custom tools + ReAct + error handling
- `streaming_dashboard.py` — FastAPI SSE endpoint
- `dual_model_optimization.py` — LIGHT + HEAVY in one agent
- `scenario_routing.py` — pre-defined scenarios with cache TTLs

Each file runs standalone in under 30 lines of setup.

### CI

`.github/workflows/ci.yml` runs `pytest`, `ruff`, and `mypy` (advisory) on
Python 3.10–3.13. A second job runs the benchmark suite in mock mode and
uploads results as a workflow artifact.

## Breaking changes

- **Minimum Python is now 3.10.** Python 3.9 reached EOL in October 2025
  and the codebase was already using 3.10+ typing features (`X | Y`,
  built-in generics) and a 3.11-only `asyncio.timeout` call. If you need
  3.9 support, pin to `swiftagentx<0.2`.

## Fixed

- `pyproject.toml` project URLs pointed to a non-existent
  `github.com/swiftagent/swiftagent` repository. Now points to
  `Caxson/swiftagentx`.
- `swiftagentx.__version__` drifted to `0.1.0` while `pyproject.toml`
  shipped `0.1.1`. Now both report `0.2.0`.
- `stream/adapter.py:event_generator_with_timeout` used the Python 3.11+
  `asyncio.timeout` context manager while the project declared
  `requires-python = ">=3.9"`. Rewritten to use a deadline check
  compatible with 3.10+.
- `tools/termination.py` silently swallowed exceptions raised by custom
  termination checkers. Exceptions are now logged at `WARNING` level
  with checker name and exception type.
- `core/memory.py:_cleanup` compared `Optional[datetime]` to `datetime`
  without a `None` guard.
- 57 ruff lint errors (18 unused imports, 38 unsorted import blocks,
  1 unused variable) auto-fixed.

## Maintainer

Author metadata corrected to point at the actual maintainer
(`Caxson <caelumsilas0@gmail.com>`). Previously listed an abstract
"SwiftAgent Team".

## Verifying this release

```bash
pip install swiftagentx==0.2.0
python -c "import swiftagentx; print(swiftagentx.__version__)"  # -> 0.2.0
git clone https://github.com/Caxson/swiftagentx.git
cd swiftagentx
pip install -e ".[dev,benchmark]"
pytest -q                                        # 105 tests, < 0.1 s
python benchmarks/run_benchmarks.py --mode mock  # reproducible numbers
```

## Next

Roadmap items being considered for v0.3.x — feedback welcome via Issues:

- Redis-backed cache backend
- Native Anthropic Claude provider (currently goes through the
  OpenAI-compatible interface)
- Concurrency / load tests in the suite
- Type-safe scenario builder (replace runtime dict with `TypedDict`)
- Hand-rolled tokenizer-aware token counting in benchmarks
