"""
Middleware system — process requests through a chain of middleware.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """
    Base middleware class.

    Implement `process` to add custom request processing logic.
    Call `await next_handler(context)` to pass to the next middleware.
    """

    @abstractmethod
    async def process(
        self,
        context: Dict[str, Any],
        next_handler: Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Process the request.

        Args:
            context: Mutable request context
            next_handler: Call this to continue to the next middleware

        Returns:
            Modified context dict
        """
        ...


class MiddlewareChain:
    """
    Chain of middleware that processes requests in order.
    """

    def __init__(self) -> None:
        self._middlewares: List[Middleware] = []

    def add(self, middleware: Middleware) -> None:
        self._middlewares.append(middleware)

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the middleware chain."""
        if not self._middlewares:
            return context

        index = 0

        async def next_handler(ctx: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal index
            if index < len(self._middlewares):
                mw = self._middlewares[index]
                index += 1
                try:
                    return await mw.process(ctx, next_handler)
                except Exception as e:
                    logger.error(f"Middleware {mw.__class__.__name__} failed: {e}", exc_info=True)
                    raise
            return ctx

        return await next_handler(context)


class TracingMiddleware(Middleware):
    """Built-in middleware for request tracing and logging."""

    async def process(
        self,
        context: Dict[str, Any],
        next_handler: Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        import time
        request_id = context.get("request_id", "unknown")
        start = time.time()
        logger.info(f"[{request_id}] Request started: {context.get('user_input', '')[:50]}")

        result = await next_handler(context)

        elapsed = (time.time() - start) * 1000
        logger.info(f"[{request_id}] Request completed in {elapsed:.1f}ms")
        result["tracing"] = {"elapsed_ms": elapsed, "request_id": request_id}
        return result
