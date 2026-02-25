"""
FastAPI SSE adapter — integrates SwiftAgent with FastAPI applications.

Requires: pip install swiftagent[fastapi]
"""

from typing import Any
import asyncio
import logging

logger = logging.getLogger(__name__)


def create_fastapi_router(agent: Any, prefix: str = "/api/v1/agent") -> Any:
    """
    Create a FastAPI APIRouter for the agent's SSE endpoint.

    Args:
        agent: SwiftAgent instance
        prefix: Route prefix

    Returns:
        FastAPI APIRouter

    Usage:
        from swiftagentx.web.fastapi_adapter import create_fastapi_router
        router = create_fastapi_router(agent)
        app.include_router(router)
    """
    try:
        from fastapi import APIRouter
        from fastapi.responses import StreamingResponse
    except ImportError:
        raise ImportError("FastAPI is required. Install with: pip install swiftagent[fastapi]")

    from ..models.schema import AgentRequest
    from ..stream.adapter import SSEStreamAdapter

    router = APIRouter(prefix=prefix, tags=["agent"])

    @router.post("/sse")
    async def sse_endpoint(request: AgentRequest) -> StreamingResponse:
        adapter = SSEStreamAdapter()

        async def run_agent_background() -> None:
            try:
                await agent.run_stream(request, adapter)
            except Exception as e:
                logger.error(f"Agent stream run failed: {e}", exc_info=True)
                from ..stream.builder import SSEEventBuilder
                error_msg = agent._sanitize_error(e) if hasattr(agent, '_sanitize_error') else "Internal error"
                await adapter.send_event(SSEEventBuilder.error(error_msg))
                await adapter.finish()

        asyncio.create_task(run_agent_background())

        return StreamingResponse(
            adapter.event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/health")
    async def health() -> dict:
        return {"status": "ok", "agent": agent.name}

    return router
