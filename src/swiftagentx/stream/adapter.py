"""
SSE stream adapter — manages event queue and generators for Server-Sent Events.
"""

import asyncio
from typing import AsyncGenerator
from datetime import datetime
from ..models.schema import StreamEvent, StreamEventType


class SSEStreamAdapter:
    """
    SSE streaming adapter.

    Sends agent execution steps to clients via Server-Sent Events protocol in real-time.
    """

    def __init__(self, buffer_size: int = 100):
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.is_finished = False
        self.events_sent: int = 0

    async def send_event(self, event: StreamEvent) -> None:
        if self.is_finished:
            raise RuntimeError("Stream is already finished")
        try:
            await asyncio.wait_for(self.event_queue.put(event), timeout=5.0)
            self.events_sent += 1
        except asyncio.TimeoutError:
            raise RuntimeError("Event queue timeout - buffer may be full")

    async def send_events(self, events: list) -> None:
        for event in events:
            await self.send_event(event)

    async def finish(self) -> None:
        self.is_finished = True
        await self.event_queue.put(None)

    async def event_generator(self) -> AsyncGenerator[str, None]:
        """Async event generator for streaming responses."""
        try:
            while True:
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=5.0)
                    if event is None:
                        break
                    yield event.to_sse_message()
                except asyncio.TimeoutError:
                    yield self._create_heartbeat()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Stream generator error: {e}", exc_info=True)
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": "Stream error occurred"},
            )
            yield error_event.to_sse_message()

    async def event_generator_with_timeout(self, timeout_seconds: int = 120) -> AsyncGenerator[str, None]:
        """Event generator with overall timeout."""
        try:
            async with asyncio.timeout(timeout_seconds):
                async for message in self.event_generator():
                    yield message
        except asyncio.TimeoutError:
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                data={"error": "Stream timeout", "timeout_seconds": timeout_seconds},
            )
            yield error_event.to_sse_message()

    @staticmethod
    def _create_heartbeat() -> str:
        heartbeat_event = StreamEvent(
            event_type=StreamEventType.INITIALIZED,
            data={"heartbeat": True, "timestamp": datetime.now().isoformat()},
        )
        return heartbeat_event.to_sse_message()

    def get_stats(self) -> dict:
        return {
            "events_sent": self.events_sent,
            "queue_size": self.event_queue.qsize(),
            "is_finished": self.is_finished,
            "buffer_size": self.buffer_size,
        }
