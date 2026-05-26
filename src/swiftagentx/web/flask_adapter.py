"""
Flask SSE adapter — integrates SwiftAgent with Flask applications.

Requires: pip install swiftagent[flask]

Note: This adapter is suitable for development and moderate traffic.
For production with high concurrency, consider using the FastAPI adapter.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

# Bounded thread pool to prevent resource exhaustion
_agent_thread_pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="swiftagent")


def create_flask_blueprint(
    agent: Any,
    url_prefix: str = "/api/v1/agent",
    max_workers: int = 32,
) -> Any:
    """
    Create a Flask Blueprint for the agent's SSE endpoint.

    Args:
        agent: SwiftAgent instance
        url_prefix: URL prefix for the blueprint
        max_workers: Max concurrent agent threads (default 32)

    Returns:
        Flask Blueprint

    Usage:
        from swiftagentx.web.flask_adapter import create_flask_blueprint
        bp = create_flask_blueprint(agent)
        app.register_blueprint(bp)
    """
    try:
        from flask import Blueprint, Response, stream_with_context
        from flask import request as flask_request
    except ImportError:
        raise ImportError("Flask is required. Install with: pip install swiftagent[flask]")

    from ..models.schema import AgentRequest
    from ..stream.adapter import SSEStreamAdapter

    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="swiftagent")
    bp = Blueprint("swiftagent", __name__, url_prefix=url_prefix)

    @bp.route("/sse", methods=["POST"])
    def sse_endpoint() -> Response:
        data = flask_request.get_json()
        if data is None:
            return Response(
                json.dumps({"error": "Invalid JSON body"}),
                status=400,
                mimetype="application/json",
            )
        agent_request = AgentRequest(**data)

        adapter = SSEStreamAdapter()

        def run_agent() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(agent.run_stream(agent_request, adapter))
            except Exception as e:
                logger.error(f"Agent run failed: {e}", exc_info=True)
            finally:
                loop.close()

        pool.submit(run_agent)

        def generate():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                gen = adapter.event_generator()
                while True:
                    try:
                        event = loop.run_until_complete(gen.__anext__())
                        yield event
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @bp.route("/health", methods=["GET"])
    def health() -> Response:
        return Response(
            json.dumps({"status": "ok", "agent": agent.name}),
            mimetype="application/json",
        )

    return bp
