# Changelog

All notable changes to SwiftAgentX are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
and uses [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [Unreleased]

## [0.3.2] — 2026-05-28

### Added

- **Scenarios now extract their template slots from natural language.**
  A `ToolChainStep(tool="weather", kwargs_template={"city": "$city"})`
  previously only worked if the caller pre-parsed the city and passed it
  as `agent.run(text, city="北京")`. Now the intent classifier extracts
  declared slots in the *same* classification call, so a user typing
  "北京天气怎么样" fires the scenario with `city=北京` automatically —
  the headline Scenario feature finally works end-to-end from a chat UI,
  still at one LLM call. New `ScenarioConfig.required_vars()` computes
  which `$slots` a scenario needs (excluding reserved keys and vars
  produced mid-chain by `extract_to`). If the classifier can't fill a
  required slot, the request gracefully falls back to ReAct instead of
  firing a step with an unsubstituted `$var`.

## [0.3.1] — 2026-05-28

A dogfood-driven patch release. Took the v0.3.0 README and tried to
build a real chatbot from it as a first-time user, then spawned four
parallel sub-agents to verify every documented feature. Six rounds of
fixes later, the same walkthrough plus all six `examples/cookbook/`
scripts, Flask, FastAPI, real-LLM benchmarks, and a literal smoke of
every Python block in the README run clean end-to-end. No new
framework features — only sharper defaults, fewer footguns, clearer
errors.

### Fixed — defaults & framework wiring

- **`Agent.run()` without `session_id` now shares one stable default
  session per Agent instance.** Previously every call generated a fresh
  random UUID, which silently disabled the v0.3 LayeredMemory feature
  for the most natural usage pattern (single Agent, multiple `await
  agent.run(text)` calls). Multi-user servers still pass an explicit
  `session_id`. (Dogfood Friction #5 — the headline bug.)
- **`Agent.run()` now accepts an `AgentRequest` polymorphically.**
  Previously `agent.run(AgentRequest(...))` crashed inside
  `_validate_input` with `TypeError: object of type 'AgentRequest' has
  no len()` — a reasonable user expectation given `run_stream(request,
  adapter)` takes one. Mixing AgentRequest + kwargs raises a clear
  `TypeError` explaining the two calling styles.
- **Three-level cache now actually writes.** `cache.set_level_2()` was
  never called from `run()` / `run_stream()` — only reads were wired.
  Cache hits worked the second time only by accident (Scenario short
  circuit's own cache layer). Now every successful turn populates L2.
- **`Agent.use(middleware)` now actually runs the chain.** The chain
  was built and middlewares were appended, but `run()` never invoked
  it — every middleware was silently dropped. Wired into both `run()`
  and `run_stream()`, with short-circuit support.
- **Six lifecycle `HookEvent`s now actually dispatch:** BEFORE / AFTER
  TOOL_CALL, BEFORE / AFTER SCENARIO_STEP, BEFORE / AFTER REACT_ITER.
  They were declared in the enum and documented but never fired —
  HookRegistry handlers attached to them would never run. Added a
  contract test (`test_every_lifecycle_hook_event_fires_at_least_once`)
  that exercises every enum value to keep this from regressing again.
- **FastAPI admin router no longer requires a double-prefix.** Users
  who followed the README pattern `app.include_router(router,
  prefix="/admin")` got endpoints at `/admin/admin/status` instead of
  `/admin/status`. The router now declares no internal prefix and
  defers entirely to the caller's mount path.
- **ReAct loop refuses to call the same tool twice in a row** with
  semantically-equivalent args. `calculator(12*34)` and `calculator(12
  * 34)` previously counted as different actions, so qwen-flash gladly
  ran them both. Dedup key now normalises whitespace inside string
  params. Real measured impact: step9_hooks_middleware latency
  5463ms → 2831ms on the same prompt.
- **Scenario engine's template substitution and `direct` output**.
  Multi-arg MCP-shaped tools were unreachable from a Scenario chain
  (only single-string `query_template` worked) — added
  `ToolChainStep.kwargs_template: dict[str, str]` so MCP `add(a, b)`-
  style tools work inside a Scenario. The `direct` output type
  branch incorrectly returned the initial `extra_vars` dict (user_id /
  session_id metadata) instead of the tool's real result when no step
  declared `extract_to`. Fixed; matches README contract.
- **SSE wire format now emits the standard `event: <type>` field**
  in front of every `data:` payload. Browser `EventSource` and
  `aiohttp-sse-client` both dispatch on this field, but it was buried
  inside the JSON payload only. README claimed "12 event types" —
  consumers couldn't actually pick which type they wanted without
  JSON-parsing every frame. Backwards-compatible — old consumers
  that only read `data:` lines still work.
- **`SSEStreamAdapter` survives a disconnected consumer**. The
  producer used to block 5s per `send_event` (and the queue's
  `put(None)` in `finish()` blocked forever) when the HTTP client
  vanished mid-stream. `put_timeout` is now 1s, the adapter
  silently no-ops further events on first timeout, and `finish()`'s
  sentinel is best-effort. Net effect: `run_stream` returns in
  ~0.1s with a dead consumer instead of deadlocking.
- **SSE answer duplication**. Streamed answers were emitted once
  during the direct/streaming path AND again at the end via
  `_stream_answer`. Now the framework tracks whether the chosen
  execution path already streamed and skips the re-emit, only
  sending `answer_end` to mark completion.
- **`LayeredMemory.flush_l2_to_l3()`** and `TopicChangeHook` now
  actually flush. The hook detected topic changes and called
  `summarize()`, but with L2 not yet overflowed L3 was empty and
  summarize silently no-op'd. Old topic kept bleeding into the new
  one via L2 replay. The hook now flushes L2 → L3 before calling
  summarize so the topic boundary actually fires.
- **`LLMHook.parse_response` tolerates real-world LLM output**. Models
  wrap JSON in `​```json … ```` fences or surround it with prose;
  parsing now tries three strategies (raw, fenced, embedded `{...}`)
  before giving up. Total parse failure logs a WARNING (was silent).
- **MCP error format**. `MCPClientError` for tools/call errors now
  reads `(code -32000): intentional server error` instead of the dict
  repr `{'code': -32000, 'message': '…'}` — easier for the LLM
  observation channel to reason about.

### Fixed — error handling & input validation

- **`OpenAICompatibleProvider` now fails fast at construction time** with
  a clear actionable message — `pip install 'swiftagentx[openai]'` — when
  `httpx` is missing, instead of crashing on the first chat() call with
  the misleading `ModuleNotFoundError: No module named 'requests'`.
  (Dogfood Friction #3.)
- **`AgentResponse.metadata` now exposes `error_class` and (only when
  `config.debug=True`) `error_message` + `traceback`** on every
  exception. Previously the user-facing answer was `"Sorry, an internal
  error occurred"` and metadata was empty. `debug=False` no longer
  leaks raw exception strings into metadata (regression caught by
  Round 3 dogfood).
- **Input-validation failures now return an `AgentResponse` with
  `metadata={"input_rejected": True, "error_class": "ValueError"}`**
  instead of raising `ValueError` out of `run()` / `run_stream()`. A
  web handler that didn't wrap the call in try/except would otherwise
  return a 500 with a leaky stack trace.
- **`StageAction` is now exported from the top-level package.** README
  pointed at `from swiftagentx import StageAction` but only
  `PipelineStage`, `RequestPipeline`, `StageResult` were re-exported —
  the example was broken.
- **Skill markdown with unclosed YAML frontmatter raises a clear
  `ValueError`** naming the missing closing `---`. Previously the
  whole frontmatter block was silently treated as body, the skill's
  `name` defaulted to its filename, and `description` was lost
  without warning.

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
- README's "LLM_API_KEY" placeholder replaced with three concrete
  provider examples (OpenAI / DashScope active / DeepSeek), so users
  can copy-paste a working configuration.
- README's Lifecycle Hooks section split into "A. Subclass Agent"
  (the 7 subclass hooks) and "B. HookRegistry" (12 declarative event
  names) so both extension patterns are discoverable.
- Both English and Chinese sections updated in lockstep.

### Tests

195 (v0.3.0) → 211 (v0.3.1). Sixteen new regression tests cover the
default session, error metadata, OpenAI provider import-error path,
ReAct duplicate-action guard, Scenario template substitution + direct
output, middleware short-circuit, FastAPI admin mounting, every
HookEvent firing, multi-kwarg ToolChainStep, disconnected SSE
consumer, input validation, and stream `send_event` post-finish
silent-drop.

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
