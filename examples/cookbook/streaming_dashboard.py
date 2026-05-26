"""
Streaming dashboard — SSE event types in action via FastAPI.

Starts a FastAPI server with a single endpoint that streams the agent's
internal events (intent classification, tool calls, observations, final
answer) as Server-Sent Events. Useful for building debug dashboards or
interactive UIs.

Run::

    pip install -e ".[fastapi]"
    uvicorn examples.cookbook.streaming_dashboard:app --reload

Then test::

    curl -N -X POST -H 'Content-Type: application/json' \\
        -d '{"user_id":"u1","session_id":"s1","user_input":"Hello!"}' \\
        http://127.0.0.1:8000/api/v1/agent/sse
"""

from __future__ import annotations

from fastapi import FastAPI

from swiftagentx import Agent, DummyModelClient, SwiftAgentConfig
from swiftagentx.web.fastapi_adapter import create_fastapi_router


def create_app() -> FastAPI:
    app = FastAPI(title="SwiftAgentX streaming dashboard")
    agent = Agent(
        model=DummyModelClient(api_key="demo", model="dummy"),
        config=SwiftAgentConfig(name="streaming-demo", sse_heartbeat_interval=5.0),
    )
    app.include_router(create_fastapi_router(agent))

    @app.get("/")
    async def root() -> dict[str, str]:
        return {
            "hint": "POST to /api/v1/agent/sse with JSON "
                    "{user_id, session_id, user_input} for streaming events",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
