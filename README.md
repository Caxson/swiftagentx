# SwiftAgentX

**A production Agent framework built around *Scenarios* — pre-compiled
execution paths that skip the ReAct loop entirely on known intents.**

[![PyPI version](https://img.shields.io/pypi/v/swiftagentx.svg)](https://pypi.org/project/swiftagentx/)
[![Python](https://img.shields.io/pypi/pyversions/swiftagentx.svg)](https://pypi.org/project/swiftagentx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/Caxson/swiftagentx/actions/workflows/ci.yml/badge.svg)](https://github.com/Caxson/swiftagentx/actions/workflows/ci.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/swiftagentx)](https://pypi.org/project/swiftagentx/)

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## The core idea: Scenarios

Other frameworks treat every request as an open-ended reasoning problem.
SwiftAgentX disagrees. In production, **80% of traffic is predictable**:
"check my order status", "what's your return policy", "book a slot at 3pm".
For these, a ReAct loop is overkill — three to five LLM calls, several
seconds of latency, a token bill that nobody can explain.

A **Scenario** is a *pre-compiled execution path*:

```python
agent.register_scenario("order_status", ScenarioConfig(
    name="Order Status",
    triggers=["order", "where is my", "shipment"],
    tool_chain=[
        ToolChainStep(tool="order_db", query_template="$order_id"),
        ToolChainStep(tool="courier_api", condition="status=in_transit"),
    ],
    cache_ttl=120,
    output_type="direct",   # no second LLM call to "format" the answer
))
```

When the LIGHT model classifies a request as a `weather` / `order_status` /
`balance_check` scenario, SwiftAgentX **executes the chain directly** —
no ReAct loop, no second LLM call. One classification step (LIGHT model,
~200 ms), one tool chain, done.

This is the framework's biggest design bet, and the place it pulls ahead of
LangChain / AutoGen / CrewAI by a margin that actually matters in
production.

## Tiered execution

Scenarios sit in the middle of a four-tier execution model:

| Request type | Path | Latency (mock LLM) | LLM calls |
|---|---|---|---|
| KB exact match / cache hit | Pipeline short-circuit | **~0 ms** | **0** |
| **Known intent (Scenario)** | **Pre-compiled tool chain** | **~200 ms** | **1** (LIGHT classify) |
| Open conversation | Direct LLM | ~1.5 s | 2 (LIGHT + HEAVY) |
| Multi-step reasoning | Full ReAct loop | 4-10 s | 3-7 |

A LIGHT model picks the path. A HEAVY model only runs when the request
genuinely needs open-ended reasoning. Measured numbers and reproducible
benchmarks live in [`benchmarks/`](benchmarks/).

### What goes inside a Scenario

A Scenario is not just a static tool list. Steps in a chain can be:

- A native Python `Tool`
- (v0.3+) An **MCP** tool — any
  [Model Context Protocol](https://modelcontextprotocol.io) server's exposed
  tools, no Python wrapper required
- (v0.3+) A **hook** — a conditional handler that branches into an LLM
  call, a sub-agent dispatch, or external shell logic when the chain hits
  a particular state

This is how Scenarios stay fast *and* extensible: the routing decision is
cheap, but each step can reach into the full agent toolkit when needed.

### vs. LangChain / AutoGen / CrewAI

|  | SwiftAgentX | LangChain | AutoGen | CrewAI |
|---|:---:|:---:|:---:|:---:|
| **Pre-compiled Scenario shortcut** | **✅ core differentiator** | ❌ no equivalent | ❌ no equivalent | ❌ no equivalent |
| FAQ / cache-hit returns with 0 LLM calls | ✅ | 1-3 LLM calls | 2+ LLM calls | 2+ LLM calls |
| Built-in three-level cache (KB / tool / session) | ✅ | partial | ❌ | ❌ |
| Dual-model routing (LIGHT/HEAVY) baked in | ✅ | DIY | DIY | DIY |
| Pipeline stage short-circuit (KB / security / feature flags) | ✅ | DIY | ❌ | ❌ |
| Streaming with fine-grained event types | ✅ 12 types | ✅ | partial | ✅ |
| Framework-agnostic core (no HTTP in `core/`) | ✅ | n/a | n/a | n/a |
| Test suite size | 105 tests, **< 0.1 s** | huge | huge | medium |

LangChain is broader. SwiftAgentX is sharper for the predictable-traffic
production patterns where latency and per-request LLM cost actually move
the needle.

## Who is this for

- You ship an Agent product where **most requests are predictable** (customer
  service, order ops, FAQ, internal copilots, AI outbound) and only a small
  tail needs real open-ended reasoning.
- You care about **P95 latency and per-request LLM cost** as first-class
  metrics, not afterthoughts.
- You want a framework you can **read in one afternoon** (4k lines of source)
  and modify without fear.
- You're comfortable wiring tools, KBs, and scenarios in Python instead of
  YAML/DSL.

If you want a kitchen-sink toolkit with every integration imaginable, use
LangChain. If you want a small, fast, opinionated core where Scenarios are
the unit of design, read on.

## Features

- **Scenarios** — Pre-compiled execution paths that skip the ReAct loop on
  known intents. The framework's headline abstraction. Each step in a
  scenario chain can be a Python tool, an MCP tool, or a conditional hook.
- **Tiered execution** — Pipeline short-circuit → Scenario → ReAct → Direct,
  picked per request by a LIGHT classifier.
- **Dual-model routing** — `ModelTier.LIGHT` for intent classification,
  `ModelTier.HEAVY` for reasoning. ~30× cost spread on real providers.
- **Three-level cache** — KB exact match (global), tool result (per-user),
  session variables. Independent TTLs, periodic cleanup.
- **Pipeline stages** — Insert KB short-circuit, security checks, feature
  flags, or any custom logic before the cache/route step. Stages can
  CONTINUE, SHORT_CIRCUIT, or ABORT.
- **Knowledge base ABC** — Built-in TF-IDF `MemoryKnowledgeBase` for local
  dev; bring your own (Weaviate, Elasticsearch, pgvector) via a 3-method ABC.
- **SSE streaming** — 12 event types (`THINKING`, `ACTION`, `OBSERVATION`,
  `ANSWER`, etc.) with heartbeats.
- **Admin API** — Status, tools, cache, config, KB endpoints as Flask
  blueprint *and* FastAPI router. Framework-agnostic core.
- **Middleware pipeline** — Tracing, retries, input validation, error
  sanitization. Hook into any stage.
- **No HTTP in core** — `httpx` is optional. You can run SwiftAgentX in
  a Lambda, a Celery worker, or a notebook.

## What's next (v0.3 roadmap)

The v0.2.0 release hardens what's already here. v0.3+ goes after the
2026-era patterns from frameworks like Claude Code:

- **MCP server support** — Scenarios and ReAct can use tools from any MCP
  server. One-line registration.
- **4-layer Memory** — Current question / last-4-turns verbatim /
  reference window / incremental rolling summary. Topic-change detection
  triggers re-summarization.
- **Hook system** — Lifecycle hooks (pre/post tool, pre/post classify) and
  semantic hooks (topic change, scenario step conditional).
- **Sub-agent dispatch** — From inside ReAct or a Scenario step, spawn a
  focused sub-agent with isolated context. Parallel dispatch supported.
- **Skill-in-ReAct** — Markdown-defined workflows the ReAct loop can pull in
  on demand (different from Scenarios, which are pre-compiled and fast).
- **Worktree-style workspace** — File sandbox per session for agents that
  generate documents.
- **Cache-friendly prompt order** — Anthropic / OpenAI prompt cache
  optimization wired into the framework.
- **Lazy tool loading** — When a registry grows past a threshold, LIGHT
  model picks the relevant category before HEAVY sees schemas.

## Installation

```bash
pip install swiftagentx
```

With optional dependencies:

```bash
pip install swiftagentx[openai]     # httpx for async OpenAI-compatible calls
pip install swiftagentx[flask]      # Flask SSE adapter
pip install swiftagentx[fastapi]    # FastAPI SSE adapter
pip install swiftagentx[all]        # Everything
```

## Quick Start

### Minimal Example

```python
import asyncio
from swiftagentx import Agent, DummyModelClient

async def main():
    agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
    response = await agent.run("Hello!")
    print(response.answer)

asyncio.run(main())
```

### With OpenAI-Compatible API

```python
import os, asyncio
from swiftagentx import Agent
from swiftagentx.providers.openai_compatible import OpenAICompatibleProvider

async def main():
    agent = Agent(
        model=OpenAICompatibleProvider(
            api_key=os.environ["LLM_API_KEY"],
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        ),
    )
    response = await agent.run("Explain quantum computing in one sentence.")
    print(response.answer)

asyncio.run(main())
```

Works with any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, DeepSeek, DashScope, etc.).

### Custom Tools

```python
from swiftagentx import Agent, Tool, ToolOutput, DummyModelClient

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(name="weather", description="Get weather for a city")

    async def execute(self, context, **kwargs):
        city = kwargs.get("city", "unknown")
        return ToolOutput(success=True, result=f"Sunny, 25C in {city}")

async def main():
    agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
    agent.register_tool(WeatherTool())
    response = await agent.run("What's the weather in Beijing?")
    print(response.answer)
```

### Dual-Model Strategy

Use a fast, cheap model for intent classification and a powerful model for reasoning:

```python
from swiftagentx import Agent, ModelTier
from swiftagentx.providers.openai_compatible import OpenAICompatibleProvider

light = OpenAICompatibleProvider(api_key=key, model="gpt-3.5-turbo", api_base=base)
heavy = OpenAICompatibleProvider(api_key=key, model="gpt-4", api_base=base)

agent = Agent(
    models={
        ModelTier.LIGHT: light,   # Intent classification (~200ms)
        ModelTier.HEAVY: heavy,   # ReAct reasoning & response generation
    },
)
```

### Scenario Toolchains

Skip the ReAct loop for common request patterns:

```python
from swiftagentx import Agent, ScenarioConfig, ToolChainStep, DummyModelClient

agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
agent.register_tool(WeatherTool())

agent.register_scenario("weather", ScenarioConfig(
    name="Weather Query",
    description="Get weather information",
    triggers=["weather", "temperature", "forecast"],
    tool_chain=[
        ToolChainStep(tool="weather", query_template="$city"),
    ],
    cache_ttl=1800,
    output_type="direct",
))
```

When the light model classifies a request as a "weather" scenario, the framework executes the tool chain directly — no ReAct loop, no extra LLM calls.

### SSE Streaming

```python
from swiftagentx import Agent, AgentRequest, SSEStreamAdapter, DummyModelClient

async def main():
    agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
    request = AgentRequest(user_id="u1", session_id="s1", user_input="Hello")
    adapter = SSEStreamAdapter()

    response = await agent.run_stream(request, adapter)

    # Events are available via adapter.event_generator()
    # In a web context, pipe this to an SSE response
```

### Knowledge Base

Attach a knowledge base to your agent. Exact matches are returned instantly, skipping LLM processing entirely:

```python
from swiftagentx import Agent, DummyModelClient, MemoryKnowledgeBase, Document

async def main():
    agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))

    kb = MemoryKnowledgeBase()
    await kb.add_documents([
        Document(doc_id="faq-1", content="Return policy: 7-day no-questions-asked returns"),
        Document(doc_id="faq-2", content="Points can be redeemed in the member store"),
    ])
    agent.set_knowledge_base(kb)  # Auto-registers KnowledgeBaseTool

    response = await agent.run("Return policy: 7-day no-questions-asked returns")
    # → Exact match (score=1.0), returned directly without LLM call
```

Use `KnowledgeBaseStage` in the pipeline for pre-processing short-circuit:

```python
from swiftagentx import KnowledgeBaseStage

agent.pipeline.add_stage(KnowledgeBaseStage(kb=kb, threshold=0.95))
```

Implement the `KnowledgeBase` ABC to integrate with Weaviate, Elasticsearch, or any vector store. See [Knowledge Base Guide](docs/guide/knowledge-base.md).

### Admin API

Monitor and manage your agent at runtime:

```python
from swiftagentx.admin import AdminService, create_flask_admin_blueprint

service = AdminService(agent)

# Flask
bp = create_flask_admin_blueprint(service)
app.register_blueprint(bp, url_prefix="/admin")

# FastAPI
from swiftagentx.admin import create_fastapi_admin_router
router = create_fastapi_admin_router(service)
app.include_router(router, prefix="/admin")
```

Available endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/admin/status` | Agent status, tool count, cache stats, uptime |
| GET | `/admin/tools` | Registered tools with JSON Schema |
| GET | `/admin/cache/stats` | Cache hit statistics |
| POST | `/admin/cache/clear` | Clear cache (all or by level) |
| GET | `/admin/config` | Current config (secrets masked) |
| PUT | `/admin/config` | Update config at runtime |
| POST | `/admin/kb/search` | Search knowledge base |
| POST | `/admin/kb/documents` | Add documents |
| DELETE | `/admin/kb/documents/:id` | Delete a document |
| GET | `/admin/kb/stats` | KB document count and provider |

> **Security**: Admin endpoints have no built-in authentication. Add your own middleware in production. See [Admin Guide](docs/guide/admin.md).

### Flask Integration

```python
from flask import Flask
from swiftagentx import Agent, DummyModelClient
from swiftagentx.web.flask_adapter import create_flask_blueprint

app = Flask(__name__)
agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
app.register_blueprint(create_flask_blueprint(agent))
# POST /api/v1/agent/sse  — SSE streaming endpoint
# GET  /api/v1/agent/health — Health check
```

### FastAPI Integration

```python
from fastapi import FastAPI
from swiftagentx import Agent, DummyModelClient
from swiftagentx.web.fastapi_adapter import create_fastapi_router

app = FastAPI()
agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
app.include_router(create_fastapi_router(agent))
```

### Lifecycle Hooks

Customize behavior without modifying framework internals:

```python
from swiftagentx import Agent

class MyAgent(Agent):
    async def on_request_start(self, context):
        print(f"Request from {context.user_id}: {context.user_input}")

    async def on_before_tool_call(self, context, tool_name, params):
        print(f"Calling tool: {tool_name}")

    async def on_before_respond(self, context, answer):
        # Modify the answer before it's sent to the user
        return answer.replace("AI", "Assistant")
```

### Middleware

```python
from swiftagentx import Agent, Middleware, DummyModelClient

class LoggingMiddleware(Middleware):
    async def process(self, context, next_handler):
        print(f"[LOG] Processing: {context.get('user_input', '')}")
        result = await next_handler(context)
        print(f"[LOG] Done")
        return result

agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
agent.use(LoggingMiddleware())
```

### Configuration

```python
from swiftagentx import Agent, SwiftAgentConfig, DummyModelClient

agent = Agent(
    model=DummyModelClient(api_key="test", model="dummy"),
    config=SwiftAgentConfig(
        name="MyAgent",
        max_iterations=5,
        enable_cache=True,
        max_input_length=5000,
        debug=False,               # Set True to expose error details
        sse_heartbeat_interval=5.0,
        max_cache_entries_per_level=10000,
    ),
)
```

## Architecture

```
User Request
    |
    v
[Middleware Chain] ──> TracingMiddleware, custom middleware, ...
    |
    v
[Pipeline Stages]
    ├─ [KnowledgeBaseStage] ─── exact match? ──> SHORT_CIRCUIT (return directly)
    ├─ [Custom Stages] ─── security check, feature flags, ...
    |
    v
[Input Validation] ─── too long? ──> Reject
    |
    v
[Cache Check] ─── hit? ──> Return cached answer (0ms)
    |
    v
[Intent Classification] (Light Model, ~200ms)
    |
    ├─ SCENARIO ──> Scenario Toolchain ──> Direct / LLM-formatted response
    ├─ REACT ────> ReAct Loop (Heavy Model) ──> Thought → Action → Observation → ... → Answer
    └─ DIRECT ───> Direct LLM Response (Heavy Model)
    |
    v
[Lifecycle Hooks] ──> on_before_respond
    |
    v
[SSE Stream / Response]
```

### Three-Level Cache

| Level | Scope | Key | TTL | Use Case |
|---|---|---|---|---|
| L1 - KB | Global | Query hash | Configurable (default 1h) | Knowledge base exact match |
| L2 - Code | Per-user + platform | User + platform + query hash | Configurable (default 5m) | Tool execution results |
| L3 - Dynamic | Per-session | Variable name | No expiry | Session state variables |
| Scenario | Per-scenario | Custom template | Configurable | Toolchain results |

## Package Structure

```
swiftagentx/
├── core/            # Agent, memory, model client, cache, prompt, parameter, router, pipeline
├── models/          # Pydantic schemas (AgentRequest, AgentResponse, config)
├── tools/           # Tool base class, registry, executor, termination checker, scenario engine
├── knowledge_base/  # KnowledgeBase ABC, MemoryKB (TF-IDF), KnowledgeBaseTool, KnowledgeBaseStage
├── admin/           # AdminService, Flask Blueprint, FastAPI Router
├── stream/          # SSE adapter and event builder
├── providers/       # LLM providers (OpenAI-compatible, DummyModelClient)
├── storage/         # Storage backend abstraction (memory, extensible)
├── middleware/       # Middleware chain (tracing, custom)
└── web/             # Web framework adapters (Flask, FastAPI)
```

## Documentation

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | System overview, dual-model strategy, cache, pipeline, ReAct loop |
| [Tools Guide](docs/guide/tools.md) | Custom tool development |
| [Scenarios Guide](docs/guide/scenarios.md) | Scenario toolchain configuration |
| [Knowledge Base Guide](docs/guide/knowledge-base.md) | KB integration, MemoryKB, custom backends |
| [Streaming Guide](docs/guide/streaming.md) | SSE events, Flask/FastAPI integration, frontend examples |
| [Admin Guide](docs/guide/admin.md) | Admin API, authentication, endpoints |
| [Deployment Guide](docs/guide/deployment.md) | Gunicorn, Uvicorn, Docker, Nginx |

## Requirements

- Python >= 3.9
- Core dependencies: `pydantic >= 2.0`, `PyYAML >= 6.0`
- No HTTP dependency in core — `httpx` is optional (for `OpenAICompatibleProvider`)

## License

Apache-2.0

---

<a id="中文"></a>

## 中文文档

# SwiftAgent

**企业级快速响应 Agent 框架**

## 为什么选择 SwiftAgent？

大多数 Agent 框架对每个请求一视同仁——扔进 ReAct 循环，调用 3-5 次 LLM。这在 Demo 阶段没问题，但在生产环境中，你需要对常见场景**亚秒级响应**，只在真正需要时才进行**深度推理**。

SwiftAgent 通过**分层执行策略**解决这个问题：

| 请求类型 | 执行路径 | 延迟 | LLM 调用次数 |
|---|---|---|---|
| 缓存命中 / KB 精准匹配 | 三级缓存 | ~0ms | 0 |
| 高频业务场景 | 场景工具链 | ~200ms | 1（仅分类） |
| 复杂推理 | 完整 ReAct 循环 | 2-10s | 3-10 |
| 简单对话 | 直接 LLM 回复 | ~500ms | 1 |

## 核心特性

- **双模型策略** — 轻量模型做意图分类（~200ms），重量模型做 ReAct 推理。分类要快，推理要深。
- **场景工具链** — 高频场景走预定义工具链，跳过 ReAct 循环，每次请求节省 2-3 次 LLM 调用。
- **三级缓存** — KB 精准匹配 / 工具结果（按用户隔离）/ 会话变量，热路径接近零延迟。
- **知识库** — 可插拔知识库抽象层，内置 TF-IDF 内存实现。支持 Pipeline 阶段精准匹配短路。通过 ABC 轻松对接 Weaviate、Elasticsearch 等向量存储。
- **管理后台** — 框架无关的管理服务层，附带 Flask Blueprint 和 FastAPI Router。开箱即用的状态、工具、缓存、配置、知识库管理端点。
- **SSE 流式** — 细粒度事件系统（思考 / 行动 / 观察 / 回答），支持心跳保活。
- **生产就绪** — 中间件流水线、请求追踪、指数退避重试、输入验证、错误脱敏。
- **框架无关** — 内置 Flask 和 FastAPI 适配器，核心零 HTTP 依赖。

## 安装

```bash
pip install swiftagentx
```

可选依赖：

```bash
pip install swiftagentx[openai]     # httpx，用于异步 OpenAI 兼容调用
pip install swiftagentx[flask]      # Flask SSE 适配器
pip install swiftagentx[fastapi]    # FastAPI SSE 适配器
pip install swiftagentx[all]        # 全部安装
```

## 快速开始

### 最简示例

```python
import asyncio
from swiftagentx import Agent, DummyModelClient

async def main():
    agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
    response = await agent.run("你好！")
    print(response.answer)

asyncio.run(main())
```

### 接入 OpenAI 兼容 API

```python
import os, asyncio
from swiftagentx import Agent
from swiftagentx.providers.openai_compatible import OpenAICompatibleProvider

async def main():
    agent = Agent(
        model=OpenAICompatibleProvider(
            api_key=os.environ["LLM_API_KEY"],
            model="gpt-4",
            api_base="https://api.openai.com/v1",
        ),
    )
    response = await agent.run("用一句话解释量子计算。")
    print(response.answer)

asyncio.run(main())
```

支持任何 OpenAI 兼容端点（OpenAI、Azure OpenAI、DeepSeek、通义千问 DashScope 等）。

### 自定义工具

```python
from swiftagentx import Agent, Tool, ToolOutput, DummyModelClient

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(name="weather", description="查询城市天气")

    async def execute(self, context, **kwargs):
        city = kwargs.get("city", "未知")
        return ToolOutput(success=True, result=f"{city}：晴，25°C")

async def main():
    agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
    agent.register_tool(WeatherTool())
    response = await agent.run("北京天气怎么样？")
    print(response.answer)
```

### 双模型策略

用快速廉价的模型做意图分类，用强力模型做推理：

```python
from swiftagentx import Agent, ModelTier
from swiftagentx.providers.openai_compatible import OpenAICompatibleProvider

light = OpenAICompatibleProvider(api_key=key, model="gpt-3.5-turbo", api_base=base)
heavy = OpenAICompatibleProvider(api_key=key, model="gpt-4", api_base=base)

agent = Agent(
    models={
        ModelTier.LIGHT: light,   # 意图分类（~200ms）
        ModelTier.HEAVY: heavy,   # ReAct 推理和回复生成
    },
)
```

### 场景工具链

跳过 ReAct 循环，直接执行预定义工具链：

```python
from swiftagentx import Agent, ScenarioConfig, ToolChainStep, DummyModelClient

agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
agent.register_tool(WeatherTool())

agent.register_scenario("weather", ScenarioConfig(
    name="天气查询",
    description="查询指定城市天气",
    triggers=["天气", "气温", "下雨"],
    tool_chain=[
        ToolChainStep(tool="weather", query_template="$city"),
    ],
    cache_ttl=1800,           # 缓存 30 分钟
    output_type="direct",     # 直接返回工具结果，无需 LLM 二次处理
))
```

当轻量模型将请求分类为 "weather" 场景时，框架直接执行工具链——不进 ReAct 循环，不产生额外 LLM 调用。

### SSE 流式响应

```python
from swiftagentx import Agent, AgentRequest, SSEStreamAdapter, DummyModelClient

async def main():
    agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
    request = AgentRequest(user_id="u1", session_id="s1", user_input="你好")
    adapter = SSEStreamAdapter()

    response = await agent.run_stream(request, adapter)
    # 事件通过 adapter.event_generator() 获取
    # 在 Web 场景中，将其接入 SSE 响应即可
```

### 知识库

为 Agent 接入知识库。精准匹配的结果直接返回，无需 LLM 处理：

```python
from swiftagentx import Agent, DummyModelClient, MemoryKnowledgeBase, Document

async def main():
    agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))

    kb = MemoryKnowledgeBase()
    await kb.add_documents([
        Document(doc_id="faq-1", content="退货政策：7天无理由退换货"),
        Document(doc_id="faq-2", content="会员积分可在商城兑换礼品"),
    ])
    agent.set_knowledge_base(kb)  # 自动注册 KnowledgeBaseTool

    response = await agent.run("退货政策：7天无理由退换货")
    # → 精准匹配 (score=1.0)，直接返回，无需 LLM 调用
```

在请求管道中使用 `KnowledgeBaseStage` 实现预处理短路：

```python
from swiftagentx import KnowledgeBaseStage

agent.pipeline.add_stage(KnowledgeBaseStage(kb=kb, threshold=0.95))
```

实现 `KnowledgeBase` ABC 即可对接 Weaviate、Elasticsearch 或任何向量存储。详见 [知识库指南](docs/guide/knowledge-base.md)。

### 管理后台

运行时监控和管理 Agent：

```python
from swiftagentx.admin import AdminService, create_flask_admin_blueprint

service = AdminService(agent)

# Flask
bp = create_flask_admin_blueprint(service)
app.register_blueprint(bp, url_prefix="/admin")

# FastAPI
from swiftagentx.admin import create_fastapi_admin_router
router = create_fastapi_admin_router(service)
app.include_router(router, prefix="/admin")
```

可用端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/admin/status` | Agent 状态、工具数、缓存统计、运行时间 |
| GET | `/admin/tools` | 已注册工具列表及 JSON Schema |
| GET | `/admin/cache/stats` | 缓存命中统计 |
| POST | `/admin/cache/clear` | 清除缓存（全部或按层级） |
| GET | `/admin/config` | 当前配置（敏感值脱敏） |
| PUT | `/admin/config` | 运行时更新配置 |
| POST | `/admin/kb/search` | 搜索知识库 |
| POST | `/admin/kb/documents` | 添加文档 |
| DELETE | `/admin/kb/documents/:id` | 删除文档 |
| GET | `/admin/kb/stats` | 知识库文档数量和提供者 |

> **安全提示**：Admin 端点不内置认证。生产环境请自行添加中间件。详见 [管理后台指南](docs/guide/admin.md)。

### Flask 集成

```python
from flask import Flask
from swiftagentx import Agent, DummyModelClient
from swiftagentx.web.flask_adapter import create_flask_blueprint

app = Flask(__name__)
agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
app.register_blueprint(create_flask_blueprint(agent))
# POST /api/v1/agent/sse   — SSE 流式端点
# GET  /api/v1/agent/health — 健康检查
```

### FastAPI 集成

```python
from fastapi import FastAPI
from swiftagentx import Agent, DummyModelClient
from swiftagentx.web.fastapi_adapter import create_fastapi_router

app = FastAPI()
agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
app.include_router(create_fastapi_router(agent))
```

### 生命周期钩子

无需修改框架源码即可自定义行为：

```python
from swiftagentx import Agent

class MyAgent(Agent):
    async def on_request_start(self, context):
        print(f"收到请求 - 用户: {context.user_id}，输入: {context.user_input}")

    async def on_before_tool_call(self, context, tool_name, params):
        print(f"调用工具: {tool_name}")

    async def on_before_respond(self, context, answer):
        # 在发送给用户之前修改回答
        return answer
```

### 中间件

```python
from swiftagentx import Agent, Middleware, DummyModelClient

class LoggingMiddleware(Middleware):
    async def process(self, context, next_handler):
        print(f"[日志] 处理请求: {context.get('user_input', '')}")
        result = await next_handler(context)
        print(f"[日志] 处理完成")
        return result

agent = Agent(model=DummyModelClient(api_key="test", model="dummy"))
agent.use(LoggingMiddleware())
```

### 配置

```python
from swiftagentx import Agent, SwiftAgentConfig, DummyModelClient

agent = Agent(
    model=DummyModelClient(api_key="test", model="dummy"),
    config=SwiftAgentConfig(
        name="MyAgent",
        max_iterations=5,          # ReAct 最大迭代次数
        enable_cache=True,         # 启用三级缓存
        max_input_length=5000,     # 输入最大长度
        debug=False,               # 生产环境设为 False，隐藏错误详情
        sse_heartbeat_interval=5.0,
        max_cache_entries_per_level=10000,
    ),
)
```

## 架构

```
用户请求
    |
    v
[中间件链] ──> TracingMiddleware, 自定义中间件, ...
    |
    v
[请求管道]
    ├─ [KnowledgeBaseStage] ─── 精准匹配? ──> 短路返回
    ├─ [自定义阶段] ─── 安全检查, 功能开关, ...
    |
    v
[输入验证] ─── 超长? ──> 拒绝
    |
    v
[缓存检查] ─── 命中? ──> 返回缓存结果 (0ms)
    |
    v
[意图分类] (轻量模型, ~200ms)
    |
    ├─ SCENARIO ──> 场景工具链 ──> 直接返回 / LLM 格式化
    ├─ REACT ────> ReAct 循环 (重量模型) ──> 思考 → 行动 → 观察 → ... → 回答
    └─ DIRECT ───> 直接 LLM 回复 (重量模型)
    |
    v
[生命周期钩子] ──> on_before_respond
    |
    v
[SSE 流式 / 响应返回]
```

### 三级缓存详解

| 层级 | 作用域 | 缓存键 | 过期策略 | 使用场景 |
|---|---|---|---|---|
| L1 - KB | 全局 | 查询哈希 | 可配置（默认 1 小时） | 知识库精准匹配 |
| L2 - Code | 按用户+平台 | 用户 + 平台 + 查询哈希 | 可配置（默认 5 分钟） | 工具执行结果 |
| L3 - Dynamic | 按会话 | 变量名 | 不过期 | 会话状态变量 |
| Scenario | 按场景 | 自定义模板 | 可配置 | 工具链结果 |

## 包结构

```
swiftagentx/
├── core/            # Agent 核心、记忆、模型客户端、缓存、提示词、参数、路由、流水线
├── models/          # Pydantic 数据模型（AgentRequest、AgentResponse、配置）
├── tools/           # 工具基类、注册表、执行器、终止检查器、场景引擎
├── knowledge_base/  # 知识库 ABC、MemoryKB（TF-IDF）、KnowledgeBaseTool、KnowledgeBaseStage
├── admin/           # AdminService、Flask Blueprint、FastAPI Router
├── stream/          # SSE 适配器和事件构建器
├── providers/       # LLM 提供者（OpenAI 兼容、DummyModelClient）
├── storage/         # 存储后端抽象（内存实现，可扩展）
├── middleware/       # 中间件链（追踪、自定义）
└── web/             # Web 框架适配器（Flask、FastAPI）
```

## 详细文档

| 文档 | 内容 |
|---|---|
| [架构总览](docs/architecture.md) | 系统架构、双模型策略、三级缓存、Pipeline、ReAct 循环 |
| [工具开发指南](docs/guide/tools.md) | 自定义工具开发 |
| [场景工具链指南](docs/guide/scenarios.md) | 场景工具链配置 |
| [知识库指南](docs/guide/knowledge-base.md) | 知识库集成、MemoryKB 用法、自定义后端 |
| [流式指南](docs/guide/streaming.md) | SSE 事件、Flask/FastAPI 集成、前端示例 |
| [管理后台指南](docs/guide/admin.md) | Admin API、认证、端点列表 |
| [部署指南](docs/guide/deployment.md) | Gunicorn、Uvicorn、Docker、Nginx |

## 环境要求

- Python >= 3.9
- 核心依赖：`pydantic >= 2.0`、`PyYAML >= 6.0`
- 核心无 HTTP 依赖 — `httpx` 为可选项（用于 `OpenAICompatibleProvider`）

## 许可证

Apache-2.0
