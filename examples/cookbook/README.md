# SwiftAgentX Cookbook

End-to-end examples for the patterns SwiftAgentX was actually designed for.
Each file is self-contained and runnable. Most use a deterministic
`DummyModelClient` so they work without an API key; the ones that need a
real LLM are marked.

## Recipes

| File | What it shows | Needs LLM? |
|------|---------------|------------|
| `customer_service_agent.py` | KB short-circuit + scenario fallback + scripted tools | No |
| `rag_chatbot.py` | Knowledge base + multi-turn dialog | No |
| `tool_calling_workflow.py` | Custom tools, ReAct loop, error handling | No |
| `streaming_dashboard.py` | SSE event types in action, FastAPI server | No (run with uvicorn) |
| `dual_model_optimization.py` | LIGHT model classifies, HEAVY model answers | Real LLM recommended |
| `scenario_routing.py` | Pre-defined toolchains skipping ReAct | No |

## How to run

```bash
pip install -e ".[dev,openai,flask,fastapi]"
python examples/cookbook/customer_service_agent.py
```

## The pattern these all share

Every recipe follows the same three-step shape:

```python
agent = Agent(model=..., config=SwiftAgentConfig(...))   # 1. construct
agent.register_tool(...) / agent.set_knowledge_base(...) # 2. wire
response = await agent.run("user input", user_id="u1")    # 3. invoke
```

If a recipe needs more than 30 lines of setup, treat that as a SwiftAgentX
bug, not a recipe problem.
