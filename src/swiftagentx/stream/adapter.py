"""
SSE stream adapter — manages event queue and generators for Server-Sent Events.
"""

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime

from ..models.schema import StreamEvent, StreamEventType


class SSEStreamAdapter:
    """
    SSE streaming adapter.

    Sends agent execution steps to clients via Server-Sent Events protocol in real-time.

    Two terminal states:

    - ``is_finished`` — the producer cleanly called :meth:`finish`. No more
      events accepted; ``event_generator`` will exit on its next ``None``.
    - ``is_closed`` — the consumer (e.g. HTTP client) disappeared and the
      queue filled. The framework switches to silent-drop mode so the
      producer doesn't keep blocking on ``queue.put`` for every step;
      subsequent ``send_event`` calls return immediately.

    ``put_timeout`` defaults to **1 second** (down from 5 s) so a single
    disconnected client costs the producer at most ~1 s of total
    blocking, not 10+ s of repeated 5 s timeouts (dogfood Friction #D-5).
    """

    def __init__(self, buffer_size: int = 100, put_timeout: float = 1.0):
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.put_timeout = put_timeout
        self.is_finished = False
        self.is_closed = False
        self.events_sent: int = 0
        self.events_dropped: int = 0

    async def send_event(self, event: StreamEvent) -> None:
        # is_closed / is_finished → silent drop. Callers don't need to
        # care whether the client is still listening; they just keep
        # producing events. This makes producer code naturally
        # cooperative with disconnects.
        if self.is_finished or self.is_closed:
            self.events_dropped += 1
            return
        try:
            await asyncio.wait_for(self.event_queue.put(event), timeout=self.put_timeout)
            self.events_sent += 1
        except (TimeoutError, asyncio.TimeoutError):
            # First put-timeout: assume the consumer is gone. Mark the
            # adapter closed so we don't keep blocking and don't keep
            # logging. Subsequent send_event calls are no-ops.
            self.is_closed = True
            self.events_dropped += 1
            import logging as _logging
            _logging.getLogger(__name__).info(
                "SSE adapter: consumer didn't read for %.1fs — marking "
                "closed; subsequent events dropped silently. "
                "(events_sent=%d, events_dropped=%d)",
                self.put_timeout, self.events_sent, self.events_dropped,
            )

    async def send_events(self, events: list) -> None:
        for event in events:
            await self.send_event(event)

    async def finish(self) -> None:
        # is_finished must flip BEFORE we touch the queue — that way any
        # producer racing with finish() takes the silent-drop branch in
        # send_event instead of also blocking on a full queue.
        self.is_finished = True
        # Sentinel delivery is best-effort. If the consumer is gone and
        # the queue is already full (is_closed path), there's nobody to
        # wake up — put_nowait raises QueueFull and we swallow it. The
        # alternative — blocking await put(None) — is the dogfood D-5
        # deadlock: 5 chunks fill the queue, send_event marks is_closed
        # and drops, then finish() blocks on put(None) forever because
        # no consumer ever drains.
        try:
            self.event_queue.put_nowait(None)
        except asyncio.QueueFull:
            self.is_closed = True
            self.events_dropped += 1

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
        """Event generator with overall timeout. Compatible with Python 3.10+."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_seconds
        try:
            async for message in self.event_generator():
                if loop.time() > deadline:
                    raise asyncio.TimeoutError()
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
