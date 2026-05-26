# Changelog

All notable changes to SwiftAgentX are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and uses [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

## [0.2.0] — 2026-05-25

This release focuses on **production readiness, observability, and benchmark
transparency**. The framework's tiered execution architecture is unchanged;
this release hardens the surrounding engineering.

### Added
- **Benchmark suite** (`benchmarks/`) with 10 scenarios across cache, scenario
  routing, ReAct, RAG, concurrency, and error recovery. Supports both a mock
  LLM mode (deterministic, CI-friendly) and a real OpenAI-compatible mode.
  Outputs per-scenario P50/P95/P99 latency, LLM call counts, and token usage,
  plus an auto-generated comparison chart.
- **Cookbook** (`examples/cookbook/`) — six runnable end-to-end examples:
  customer service agent, RAG chatbot, tool-calling workflow, streaming
  dashboard, dual-model optimization, and scenario routing.
- `pytest-cov` and a coverage configuration in `pyproject.toml`.
- `benchmark` optional dependency group (`pip install swiftagentx[benchmark]`)
  bundling `matplotlib` and `tabulate`.
- `Issues` and `Changelog` URLs to `pyproject.toml`.
- GitHub Actions CI workflow running tests on Python 3.10 / 3.11 / 3.12 /
  3.13 plus `ruff` and `mypy` gates.
- `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.

### Changed
- **Minimum Python version raised to 3.10** (3.9 reached EOL in October 2025
  and the codebase already used 3.10-only typing features).
- `pyproject.toml` ruff rules extended with `B` (bugbear) and `UP` (pyupgrade);
  target bumped to `py310`.
- `Development Status` classifier moved from `3 - Alpha` to `4 - Beta`.
- README rewritten with a sharper positioning, head-to-head comparison against
  LangChain, and embedded benchmark results.
- Author metadata corrected to point at the actual maintainer.

### Fixed
- **Pipeline stages were never executed.** `Agent.pipeline` was created in
  `__init__` but `Agent.run()` skipped the pipeline entirely, so
  `agent.pipeline.add_stage(...)` was a silent no-op. The README documented
  `KnowledgeBaseStage` as a way to get exact-match short-circuits — that
  promise was broken. Now `run()` executes all pipeline stages before the
  cache check, and a stage returning `SHORT_CIRCUIT` returns immediately
  without any LLM call. Tests added in `tests/test_pipeline_integration.py`.
- **`set_knowledge_base` now auto-installs a `KnowledgeBaseStage`** (opt-out
  with `auto_short_circuit=False`). Previously, calling
  `agent.set_knowledge_base(kb)` only registered a `KnowledgeBaseTool` for
  use inside the ReAct loop; users had to remember to also add a pipeline
  stage to get the documented "zero LLM calls on exact match" behavior.
  Calling `set_knowledge_base` twice is now idempotent.
- Version mismatch between `pyproject.toml` (`0.1.1`) and `swiftagentx.__version__`
  (`0.1.0`).
- `pyproject.toml` project URLs previously pointed to a non-existent
  `github.com/swiftagent/swiftagent` repository.
- `stream/adapter.py` used `asyncio.timeout()` (Python 3.11+) while
  declaring `requires-python = ">=3.9"`. Rewritten with a `wait_for`-based
  approach compatible with Python 3.10+.
- `tools/termination.py` silently swallowed exceptions raised by custom
  termination checkers; now logged at `WARNING` with checker name and exception
  type.
- `core/memory.py` cleanup compared `Optional[datetime]` against `datetime`
  without a `None` guard.
- Removed 18 unused imports and reformatted 38 import blocks via `ruff --fix`.

### Removed
- Empty `examples/customer_service/` placeholder directory replaced by
  `examples/cookbook/customer_service_agent.py`.

## [0.1.1] — 2026-02-25

Initial public release on PyPI.

### Added
- Core `Agent` with tiered execution: cache hit → scenario toolchain →
  ReAct loop → direct LLM response.
- Three-level cache (`CacheManager`): global KB, per-user tool result,
  per-session dynamic variables.
- Dual-model abstraction (`ModelTier.LIGHT` / `ModelTier.HEAVY`).
- Pluggable `KnowledgeBase` ABC with built-in TF-IDF `MemoryKnowledgeBase`.
- `KnowledgeBaseStage` pipeline stage for exact-match short-circuit.
- SSE streaming adapter with twelve event types.
- Flask and FastAPI adapters; framework-agnostic `AdminService`.
- Middleware chain with built-in `TracingMiddleware`.
- 105 tests covering core, cache, KB, admin, tools, streaming.

[Unreleased]: https://github.com/Caxson/swiftagentx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Caxson/swiftagentx/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Caxson/swiftagentx/releases/tag/v0.1.1
