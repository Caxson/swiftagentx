# SSE 流式指南

## 概述

SwiftAgent 通过 Server-Sent Events (SSE) 提供细粒度的流式事件，让前端实时展示 Agent 的思考过程、工具调用和最终回复。

## 事件类型

| 事件 | 说明 |
|------|------|
| `INITIALIZED` | Agent 就绪 |
| `KB_LOOKUP` | 知识库查询 |
| `CACHE_HIT` | 缓存命中 |
| `THOUGHT_START/CHUNK/END` | 思考过程 |
| `ACTION_START/CHUNK/END` | 工具选择 |
| `OBSERVATION` | 工具执行结果 |
| `ANSWER_START/CHUNK/END` | 最终回复 |
| `MESSAGE_END` | 消息元数据 |
| `COMPLETED` | 请求完成 |
| `ERROR` | 错误 |

## SSEStreamAdapter

```python
from swiftagentx import SSEStreamAdapter

adapter = SSEStreamAdapter(buffer_size=100)

# 发送事件
await adapter.send_event(event)

# 完成
await adapter.finish()

# 消费事件（异步生成器）
async for sse_line in adapter.event_generator():
    yield sse_line

# 带超时的消费
async for sse_line in adapter.event_generator_with_timeout(timeout_seconds=120):
    yield sse_line
```

## SSEEventBuilder

便捷工厂类，创建各类事件：

```python
from swiftagentx import SSEEventBuilder

SSEEventBuilder.initialized("Agent ready")
SSEEventBuilder.thought_start(iteration=1)
SSEEventBuilder.thought_end("I need to check the weather", iteration=1)
SSEEventBuilder.action_start("get_weather", iteration=1)
SSEEventBuilder.action_end("get_weather", {"city": "Beijing"}, iteration=1)
SSEEventBuilder.observation("Beijing: 晴天 25°C", iteration=1)
SSEEventBuilder.answer_chunk("今天北京天气晴朗...")
SSEEventBuilder.completed()
SSEEventBuilder.error("Something went wrong")
```

## Flask 集成

```python
from flask import Flask, Response
from swiftagentx import Agent, AgentRequest, SSEStreamAdapter

app = Flask(__name__)
agent = Agent(model=my_model)

@app.route("/api/v1/agent/sse", methods=["POST"])
def agent_sse():
    data = request.get_json()
    req = AgentRequest(**data)
    adapter = SSEStreamAdapter()

    async def process():
        await agent.run_stream(req, adapter)

    import asyncio
    loop = asyncio.new_event_loop()
    loop.run_in_executor(None, lambda: asyncio.run(process()))

    def generate():
        import asyncio as aio
        loop2 = aio.new_event_loop()
        gen = adapter.event_generator_with_timeout(120)
        while True:
            try:
                line = loop2.run_until_complete(gen.__anext__())
                yield line
            except StopAsyncIteration:
                break

    return Response(generate(), mimetype="text/event-stream")
```

> SwiftAgent 的 `web/flask_adapter.py` 提供了更完整的集成方案。

## FastAPI 集成

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from swiftagentx import Agent, AgentRequest, SSEStreamAdapter

app = FastAPI()
agent = Agent(model=my_model)

@app.post("/api/v1/agent/sse")
async def agent_sse(req: AgentRequest):
    adapter = SSEStreamAdapter()

    import asyncio
    asyncio.create_task(agent.run_stream(req, adapter))

    return StreamingResponse(
        adapter.event_generator_with_timeout(120),
        media_type="text/event-stream",
    )
```

## 前端示例

```html
<script>
const eventSource = new EventSource('/api/v1/agent/sse');

eventSource.addEventListener('message', (e) => {
    const data = JSON.parse(e.data);

    switch (data.event_type) {
        case 'thought_end':
            console.log('Thought:', data.data.full_thought);
            break;
        case 'action_start':
            console.log('Calling tool:', data.data.tool_name);
            break;
        case 'answer_chunk':
            document.getElementById('answer').textContent += data.data.answer;
            break;
        case 'completed':
            eventSource.close();
            break;
        case 'error':
            console.error('Error:', data.data.error);
            eventSource.close();
            break;
    }
});
</script>
```

也可以使用 `fetch` + `ReadableStream` 处理 POST 请求的 SSE：

```javascript
const response = await fetch('/api/v1/agent/sse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        user_id: 'user1',
        session_id: 'sess1',
        user_input: '查询积分余额',
    }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    // 解析 SSE 行
    for (const line of text.split('\n')) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            handleEvent(data);
        }
    }
}
```
