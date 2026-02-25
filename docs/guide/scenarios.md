# 场景工具链指南

## 概述

场景工具链（Scenario Toolchain）允许你为高频场景预定义工具调用序列，跳过完整的 ReAct 推理循环，实现更快的响应。

## 快速开始

```python
from swiftagentx import Agent, ScenarioConfig, ToolChainStep

agent = Agent(model=my_model)

# 定义场景
scenario = ScenarioConfig(
    name="查询积分",
    description="查询用户积分余额",
    triggers=["积分", "points", "余额"],
    tool_chain=[
        ToolChainStep(tool="get_user_info", extract_to="user"),
        ToolChainStep(tool="get_points", query_template="{user.id}"),
    ],
    cache_key_template="{user_id}:points",
    cache_ttl=300,
)

agent.register_scenario("points_query", scenario)
```

## ScenarioConfig

```python
class ScenarioConfig(BaseModel):
    name: str                        # 场景名称
    description: str = ""            # 描述
    triggers: List[str] = []         # 触发关键词（用于意图分类）
    tool_chain: List[ToolChainStep]  # 工具链步骤
    cache_key_template: str = ""     # 缓存键模板
    cache_ttl: int = 3600            # 缓存 TTL（秒）
    output_template: str = "llm"     # 输出模板（"llm" 或自定义模板）
    output_type: str = "llm_processed"
```

## ToolChainStep

```python
class ToolChainStep(BaseModel):
    tool: str                    # 工具名称
    extract_to: str = ""         # 将结果存储到变量名
    condition: str = "always"    # 执行条件
    query_template: str = ""     # 参数模板（支持 {var} 替换）
```

### 步骤执行流程

```
Step 1: get_user_info → 结果存入 context["user"]
    ↓
Step 2: get_points(user_id=context["user"]["id"]) → 结果存入 context["points"]
    ↓
Output: LLM 格式化结果 / 直接输出模板
```

## 缓存模板

`cache_key_template` 支持变量替换：

```python
# 按用户缓存
cache_key_template="{user_id}:points"

# 按用户+平台缓存
cache_key_template="{user_id}:{platform}:order"
```

## 输出类型

| output_type | 行为 |
|-------------|------|
| `llm_processed` | 使用 HEAVY 模型将工具结果加工为自然语言回复 |
| `direct_output` | 直接返回工具链最后一步的结果 |
| `script_output` | 使用 output_template 格式化输出 |

## 完整示例

```python
from swiftagentx import (
    Agent, Tool, ToolOutput, ScenarioConfig, ToolChainStep,
    DummyModelClient,
)

# 1. 定义工具
class GetUserTool(Tool):
    def __init__(self):
        super().__init__(name="get_user", description="获取用户信息")

    async def execute(self, context, **kwargs):
        user_id = context.variables.get("user_id", "unknown")
        return ToolOutput(success=True, result={"id": user_id, "name": "张三"})

class GetPointsTool(Tool):
    def __init__(self):
        super().__init__(name="get_points", description="查询积分")

    async def execute(self, context, **kwargs):
        return ToolOutput(success=True, result={"balance": 1500, "level": "gold"})

# 2. 创建 Agent 并注册
agent = Agent(model=DummyModelClient(api_key="k", model="m"))
agent.register_tool(GetUserTool())
agent.register_tool(GetPointsTool())

# 3. 注册场景
agent.register_scenario("check_points", ScenarioConfig(
    name="查积分",
    triggers=["积分", "余额"],
    tool_chain=[
        ToolChainStep(tool="get_user", extract_to="user"),
        ToolChainStep(tool="get_points", extract_to="points"),
    ],
))
```
