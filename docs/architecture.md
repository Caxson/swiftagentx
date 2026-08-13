# SwiftAgent 架构总览

SwiftAgent 是一个企业级快速响应 Agent 框架，提供 dual-model 策略、三级缓存、场景工具链、SSE 流式和 ReAct 推理循环。

## 系统架构图

```
┌──────────────────────────────────────────────────────────┐
│                     Client (Web/App)                     │
│                                                          │
│  SSE EventSource  ←─── /api/v1/agent/sse ──→  Flask/    │
│  JSON Response    ←─── /api/v1/agent/chat ──→  FastAPI  │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│                   Middleware Chain                        │
│                                                          │
│  TracingMiddleware → CustomMiddleware → ...               │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│                   Request Pipeline                       │
│                                                          │
│  [KnowledgeBaseStage] → [CacheStage] → [CustomStage]    │
│         ↓ SHORT_CIRCUIT (exact match)                    │
│         ↓ CONTINUE (carry kb_results)                    │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│                    Agent Core                            │
│                                                          │
│  ┌─────────────┐    ┌──────────────────────┐             │
│  │ Light Model │───→│  Intent Router       │             │
│  │ (qwen-mini) │    │  classify(input)     │             │
│  └─────────────┘    └───────┬──────────────┘             │
│                       ┌─────┴─────┐                      │
│                 Level1│   Level2  │Level3                 │
│                 REACT │  SCENARIO │DIRECT                 │
│                   │   │     │     │  │                    │
│  ┌────────────────▼─┐ │ ┌───▼───┐│  ▼                    │
│  │  ReAct Loop      │ │ │Scenario│ Direct LLM            │
│  │  (heavy model)   │ │ │Engine │ Response               │
│  │                  │ │ └───────┘│                        │
│  │  Thought→Action  │ │         │                        │
│  │  →Observation    │ │         │                        │
│  │  → ... → Answer  │ │         │                        │
│  └──────────────────┘ │         │                        │
│                       └─────────┘                        │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│                    Tool System                           │
│                                                          │
│  ToolRegistry ──→ ToolExecutor (retry + timeout)         │
│      │                                                   │
│      ├── KnowledgeBaseTool                               │
│      ├── CustomTool1                                     │
│      └── CustomTool2                                     │
└──────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│                  Three-Level Cache                        │
│                                                          │
│  Level 1: KB Cache     (exact match, TTL=3600s)          │
│  Level 2: Tool Cache   (per-user, TTL=300s)              │
│  Level 3: Session Cache (no expiry)                      │
│  Scenario: Toolchain Cache (configurable TTL)            │
└──────────────────────────────────────────────────────────┘
```

## 核心概念

### 1. 双模型策略 (Dual-Model Strategy)

SwiftAgent 支持两个模型层级：

| 层级 | 用途 | 特点 |
|------|------|------|
| **LIGHT** | 意图分类、快速判断 | 低延迟 (~200ms) |
| **HEAVY** | 任务执行、ReAct 推理、回复生成 | 高质量 |

```python
from swiftagentx import Agent, ModelTier

agent = Agent(
    models={
        ModelTier.LIGHT: light_model,   # 用于分类
        ModelTier.HEAVY: heavy_model,   # 用于推理
    },
)
```

如果只提供一个 `model` 参数，它会同时用于 LIGHT 和 HEAVY。

### 2. 三级缓存 (Three-Level Cache)

缓存按优先级查询：

1. **Level 1 — KB Cache**: 知识库精准匹配结果，长 TTL (默认 3600s)
2. **Level 2 — Tool Cache**: 工具调用结果，按 user_id 隔离，短 TTL (默认 300s)
3. **Level 3 — Session Cache**: 会话级变量，无过期时间

### 3. 意图分类与路由 (Intent Classification)

IntentRouter 使用 LIGHT 模型将用户输入分为三个级别：

| Level | 名称 | 行为 |
|-------|------|------|
| 1 | REACT | 进入 ReAct 推理循环（多步工具调用） |
| 2 | SCENARIO | 执行预定义场景工具链 |
| 3 | DIRECT | 直接 LLM 回复（无需工具） |

### 4. ReAct 推理循环

```
用户输入 → Thought (推理) → Action (工具选择) → Observation (结果)
    ↑                                                        │
    └────────────────────────────────────────────────────────┘
    (循环直到找到 Final Answer 或达到 max_iterations)
```

### 5. 场景工具链 (Scenario Toolchains)

对高频场景跳过完整的 ReAct 循环，直接按预定义步骤执行：

```python
from swiftagentx import ScenarioConfig, ToolChainStep

scenario = ScenarioConfig(
    name="查询积分",
    triggers=["积分", "points"],
    tool_chain=[
        ToolChainStep(tool="get_user_info", extract_to="user"),
        ToolChainStep(tool="get_points", query_template="{user.id}"),
    ],
)
agent.register_scenario("points_query", scenario)
```

### 6. SSE 流式事件

Agent 通过 `SSEStreamAdapter` 发送细粒度事件：

```
INITIALIZED → THOUGHT_START → THOUGHT_END → ACTION_START → ACTION_END
→ OBSERVATION → ... → ANSWER_START → ANSWER_CHUNK → ANSWER_END → COMPLETED
```

### 7. 请求管道 (Request Pipeline)

可插拔的阶段链，用于请求预处理：

```python
from swiftagentx import KnowledgeBaseStage

agent.pipeline.add_stage(KnowledgeBaseStage(kb=my_kb))
```

每个阶段返回 `CONTINUE`（继续）、`SHORT_CIRCUIT`（直接返回）或 `ABORT`（终止）。

### 8. 生命周期钩子

继承 `Agent` 并覆盖钩子方法：

```python
class MyAgent(Agent):
    async def on_request_start(self, context): ...
    async def on_before_classify(self, context): ...
    async def on_after_classify(self, context, result): ...
    async def on_before_tool_call(self, context, tool_name, params): ...
    async def on_after_tool_call(self, context, tool_name, result): ...
    async def on_before_respond(self, context, answer) -> str: ...
    async def on_request_end(self, context, response): ...
```

### 9. Scenario 引擎进阶能力（daily-opt D1–D8）

`tools/scenario.py` 的 `ScenarioEngine` 在上面的基础执行模型上叠加了几层
能力，`core/agent.py` 按需接入，全部默认关闭/透明，不改变未开启时的行为：

- **并行步骤组**：`tool_chain` 的一项可以是 `list[ToolChainStep]`（无
  依赖 step 组），`asyncio.gather` fan-out 后 join；`on_group_failure`
  控制组内部分失败是 `fail_fast` 还是 `best_effort`。
- **断点续跑**：`ScenarioCheckpoint` 每个组 join 后落盘到 `Workspace`，
  进程重启后可从最后完成的分组恢复。
- **Context 卸载**：超阈值 tool 输出与大号 `direct` 答案只写一次
  workspace，prompt 里留引用 + 摘要，支持分块回读。
- **候选挖矿 + 回放评测 + 自动转正**：`core/miner.py` 把 ReAct 实跑的
  工具链聚类成 Planner 候选，`core/replay_eval.py` 用历史请求重放候选
  并与 ReAct 基线打分，两者共用 `core/planner.py` 的 `PlanStore` 复用/
  转正闸门；`plan_promote_requires_eval` 可要求自动转正必须先过评测。

详细设计与验证方式见 `docs/OPTIMIZATION_PLAN.md`（D1–D8）。

## 模块依赖关系

```
swiftagentx/
├── core/           # Agent, Memory, Cache, Router, Pipeline, Prompt
├── tools/          # Tool ABC, Registry, Executor, Scenario Engine
├── models/         # Config, Schema (request/response/events)
├── stream/         # SSE Adapter, Event Builder
├── middleware/      # Middleware chain
├── knowledge_base/ # KnowledgeBase ABC, MemoryKB, Tool, Stage
├── admin/          # AdminService, Flask/FastAPI routes
├── storage/        # Storage backend abstraction
├── providers/      # LLM provider implementations
└── web/            # Web framework adapters
```

## 数据流

```
AgentRequest
    → Middleware Chain
    → Pipeline Stages (KB lookup, cache check, ...)
    → Intent Classification (Light Model)
    → Execution (ReAct / Scenario / Direct)
    → Post-processing hooks
    → AgentResponse (+ SSE events if streaming)
```
