# Changelog

All notable changes to SwiftAgentX are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and uses [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

## [0.3.1] — 2026-05-27

A dogfood-driven patch release. Five frictions surfaced when actually
building a CLI chatbot end-to-end from the v0.3.0 README; this release
fixes all of them. No new framework features — only sharper defaults
and clearer docs.

### Fixed

- **`Agent.run()` without `session_id` now shares one stable default
  session per Agent instance.** Previously every call generated a fresh
  random UUID, which silently disabled the v0.3 LayeredMemory feature
  for the most natural usage pattern (single Agent, multiple `await
  agent.run(text)` calls). Multi-user servers still pass an explicit
  `session_id`. (Dogfood Friction #5 — the headline bug.)
- **`OpenAICompatibleProvider` now fails fast at construction time** with
  a clear actionable message — `pip install 'swiftagentx[openai]'` — when
  `httpx` is missing, instead of crashing on the first chat() call with
  the misleading `ModuleNotFoundError: No module named 'requests'`.
  (Dogfood Friction #3.)
- **`AgentResponse.metadata` now exposes `error_class` and
  `error_message`** on every exception, even when `config.debug=False`.
  Previously the user-facing answer was `"Sorry, an internal error
  occurred"` and metadata was an empty dict — leaving callers no way to
  branch on the failure mode. With `debug=True`, a full traceback is
  also attached as `metadata["traceback"]`. (Dogfood Friction #4.)
- **README's top-of-file benchmark reproduce command now prefixes with
  `git clone` + `pip install -e ".[dev,openai,benchmark]"`** so a fresh
  reader can actually run it. The previous version expected the
  repo to already be local. (Dogfood Friction #1.)

### Changed

- The `[openai]` extra now installs `httpx[socks]>=0.25.0` instead of
  plain `httpx`. This pulls in `socksio` so users behind a SOCKS proxy
  (common in mainland China deployments — explicitly called out in the
  project's `CLAUDE.md` policies) don't crash on the first request with
  `ImportError: Using SOCKS proxy, but the 'socksio' package is not
  installed`.

### Docs

- README's "OpenAI-Compatible API" Quick Start now prefaces with the
  required extras install and the China-mainland proxy gotcha — the
  two things that block a new user inside 60 seconds.
- README gains a "Multi-turn conversations" section explicitly showing
  the default-session pattern and when to pass an explicit `session_id`.
- Both English and Chinese sections updated in lockstep.

### Tests

195 (v0.3.0) → 201 (v0.3.1). Six new regression tests in
`tests/test_default_session.py` cover the default session, error
metadata behavior, and OpenAI provider import-error path.

## [0.3.0] — 2026-05-26

This release brings the framework into the 2026 generation of agent patterns
(memory, hooks, MCP, sub-agents, skills) while keeping **Scenarios as the
headline abstraction**. Every new subsystem is a building block a Scenario
or ReAct iteration can use — Scenarios are not replaced.

### Added

- **4-layer Memory** (`LayeredMemory`, `LayeredMemoryStore`) — L1 current
  question / L2 last-4-turns verbatim / L3 reference window / L4 incremental
  rolling summary. Cadence-based (every N turns) and semantic-hook-triggered
  summarization paths both exist. Pluggable `MemoryBackend`; ships with
  `InMemoryBackend` (production-ready future: Redis backend).
- **Hook system** (`HookRegistry`, `HookEvent`) — 12 lifecycle events and
  4 semantic events with four handler kinds (`PythonHook`, `LLMHook`,
  `ShellHook`, semantic hooks). The v0.2 subclass-override pattern still
  works alongside.
- **`TopicChangeHook`** — built-in semantic hook that asks the LIGHT model
  whether the current input starts a new topic; on detection, calls
  `memory.summarize()` so the layered memory stays coherent across topic
  switches. Auto-registered; opt-out via config.
- **MCP server support** (`Agent.register_mcp_server`, `MCPClient`,
  `MCPTool`) — Scenarios and ReAct can call any Model Context Protocol
  server's tools by name. Stdio + SSE transports.
- **Sub-agent dispatch** (`SubAgentRole`, `Agent.dispatch_subagents`) —
  parallel focused agents with isolated context, structured results, and
  one-failed-doesn't-break-others fan-out.
- **Skill-in-ReAct** (`Skill`, `Agent.invoke_skill`, `Agent.load_skills`) —
  markdown-defined workflows the ReAct loop can invoke. Complement to
  Scenarios; not a replacement.
- **Session workspace** (`Workspace`, `Agent.workspace`) — per-session file
  sandbox with `LocalDiskWorkspaceBackend` + `InMemoryWorkspaceBackend`,
  path-escape protection, optional cleanup-on-exit.
- **Cache-friendly prompt layout** (`PromptLayout`) — assembles prompts in
  least-changing → most-changing order (tools → system → L4 → L3 → L2 → L1)
  for Anthropic/OpenAI prompt-cache friendliness.
- **Lazy tool loading** (`ToolRegistry.select_tools_for_query`,
  `schemas_for_query`) — when a registry exceeds a threshold, score tools
  against the query and return only the top-K. Important when many MCP
  servers contribute hundreds of tools.
- **Real-LLM benchmark runner** (`benchmarks/real_runner.py`) — exercises
  the four execution tiers against any OpenAI-compatible endpoint
  (defaults to DashScope qwen-flash + qwen-turbo), emits JSON + matplotlib
  chart. 30 iterations per scenario costs well under one yuan.
- **`docs/architecture-v0.3.md`** — binding construction blueprint for the
  release, including OUT-of-scope items (no permissions, slash commands,
  CLI, output styles, dashboards).
- README headline visual embeds the measured benchmark chart from
  `docs/assets/v0.3-benchmark-qwen.png`.

### Changed

- `Agent.memory` is now a `LayeredMemoryStore` (per-session multiplexer)
  instead of the singleton `SessionMemory` that pooled every session's
  history together (latent v0.2 bug — sessions could see each other's
  context). The standalone `SessionMemory` class itself is unchanged for
  users who construct it directly.
- `Agent.run()` / `Agent.run_stream()` now dispatch lifecycle hooks at
  every boundary in addition to calling the subclass-override methods
  (`on_request_start` etc.). The two paths are additive.
- `_direct_response()` now injects layered memory into the chat prompt
  via `mem.to_chat_messages(l2_rounds=5)` rather than the old
  `get_conversation_for_reply`.
- README test-suite count updated from 111 to 195. Tiered-execution table
  replaced with measured P50/P95 numbers from the new benchmark runner.

### Breaking changes

- The undocumented patterns `agent.memory.add_message(...)` and
  `agent.memory.get_conversation_for_reply(...)` no longer work on the
  agent attribute (it's no longer a `SessionMemory`). Importing the
  `SessionMemory` class directly from `swiftagentx.core.memory` still
  works as before.

### Numbers (measured, not aspirational)

DashScope Qwen, 30 iterations per scenario, LIGHT=`qwen-flash`,
HEAVY=`qwen-turbo`:

| Tier               | P50    | P95    | LLM calls |
|--------------------|-------:|-------:|----------:|
| KB exact match     |   0 ms |   0 ms |         0 |
| Scenario shortcut  | 517 ms | 802 ms |         1 |
| Cache hit          |   0 ms |   0 ms |         0 |
| Simple QA (DIRECT) |  1.4 s |  2.4 s |         2 |
| ReAct complex      |  3.1 s |  4.0 s |         3 |

Reproduce: `python benchmarks/real_runner.py --iterations 30`.

### Tests

105 (v0.2.0) → 195 (v0.3.0). Full suite runs in <0.5 s.

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

[Unreleased]: https://github.com/Caxson/swiftagentx/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/Caxson/swiftagentx/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Caxson/swiftagentx/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Caxson/swiftagentx/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Caxson/swiftagentx/releases/tag/v0.1.1
