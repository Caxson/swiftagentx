# 自定义工具开发指南

## 概述

Tool 是 SwiftAgent 与外部系统交互的基本单元。Agent 在 ReAct 循环中调用工具获取信息或执行操作。

## 快速开始

```python
from swiftagentx import Tool, ToolOutput, ToolOutputType, AgentContext

class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather for a city",
            category="utility",
        )

    async def execute(self, context: AgentContext, **kwargs) -> ToolOutput:
        city = kwargs.get("city", "")
        # ... 调用天气 API ...
        return ToolOutput(success=True, result=f"{city}: 晴天, 25°C")

    def get_schema(self) -> dict:
        schema = super().get_schema()
        schema["parameters"] = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
            },
            "required": ["city"],
        }
        return schema
```

注册到 Agent：

```python
agent.register_tool(WeatherTool())
```

## Tool 基类

```python
class Tool(ABC):
    def __init__(
        self,
        name: str,              # 工具名称（唯一标识）
        description: str,       # 描述（LLM 用于理解工具功能）
        category: str = "general",
        output_type: ToolOutputType = ToolOutputType.LLM_PROCESSED,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ): ...

    @abstractmethod
    async def execute(self, context: AgentContext, **kwargs) -> ToolOutput: ...

    def validate_input(self, **kwargs) -> bool: ...
    def get_schema(self) -> Dict[str, Any]: ...
```

## ToolOutput

```python
class ToolOutput(BaseModel):
    success: bool           # 是否成功
    result: Any             # 返回结果
    error: Optional[str]    # 错误信息
    output_type: ToolOutputType  # 输出类型
    metadata: Dict[str, Any]     # 元数据
```

### 输出类型

| 类型 | 说明 |
|------|------|
| `LLM_PROCESSED` | 结果需要 LLM 加工后回复用户（默认） |
| `DIRECT_OUTPUT` | 直接返回给用户，跳过 LLM |
| `SCRIPT_OUTPUT` | 已格式化的输出 |

## AgentContext

工具接收的上下文协议：

```python
class AgentContext(Protocol):
    variables: Dict[str, Any]  # 会话变量
    user_input: str            # 用户输入
    session_id: str            # 会话 ID
```

`SessionContext` 是内置的实现，也可以传入自定义对象。

## JSON Schema

`get_schema()` 返回的 JSON Schema 用于 LLM 的 function calling：

```python
def get_schema(self) -> dict:
    schema = super().get_schema()
    schema["parameters"] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数量",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    return schema
```

## 输入验证

```python
class MyTool(Tool):
    def validate_input(self, **kwargs) -> bool:
        return "query" in kwargs and len(kwargs["query"]) > 0
```

## 错误处理

工具执行由 `ToolExecutor` 包装，自动支持：
- **超时**: 默认 30 秒
- **重试**: 默认最多 3 次，指数退避
- **错误捕获**: 异常转为 `ToolOutput(success=False, error=...)`

## 完整示例：数据库查询工具

```python
class DatabaseQueryTool(Tool):
    def __init__(self, db_connection):
        super().__init__(
            name="query_db",
            description="Query the database for user information",
            category="database",
            timeout_seconds=15,
        )
        self.db = db_connection

    async def execute(self, context: AgentContext, **kwargs) -> ToolOutput:
        sql = kwargs.get("sql", "")
        if not sql:
            return ToolOutput(success=False, result=None, error="SQL query required")

        try:
            results = await self.db.execute(sql)
            return ToolOutput(success=True, result=results)
        except Exception as e:
            return ToolOutput(success=False, result=None, error=str(e))

    def get_schema(self) -> dict:
        schema = super().get_schema()
        schema["parameters"] = {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL 查询语句"},
            },
            "required": ["sql"],
        }
        return schema
```
