"""
Tests for the SSE streaming system.
"""


import pytest

from swiftagentx import SSEEventBuilder, SSEStreamAdapter
from swiftagentx.models.schema import StreamEventType


class TestSSEStreamAdapter:
    @pytest.mark.asyncio
    async def test_send_and_receive_event(self):
        adapter = SSEStreamAdapter(buffer_size=10)
        event = SSEEventBuilder.initialized("test init")
        await adapter.send_event(event)
        await adapter.finish()

        events = []
        async for msg in adapter.event_generator():
            events.append(msg)

        assert len(events) == 1
        assert "initialized" in events[0]

    @pytest.mark.asyncio
    async def test_finish_sets_flag(self):
        adapter = SSEStreamAdapter()
        assert not adapter.is_finished
        await adapter.finish()
        assert adapter.is_finished

    @pytest.mark.asyncio
    async def test_send_after_finish_drops_silently(self):
        """v0.3.1+ contract: send_event after finish/close drops the event
        instead of raising. Producers don't need try/except around every
        send_event call — that's cooperative cancellation for SSE."""
        adapter = SSEStreamAdapter()
        await adapter.finish()
        # Must NOT raise.
        await adapter.send_event(SSEEventBuilder.initialized())
        assert adapter.events_dropped >= 1
        assert adapter.is_finished

    def test_stats(self):
        adapter = SSEStreamAdapter(buffer_size=50)
        stats = adapter.get_stats()
        assert stats["events_sent"] == 0
        assert stats["buffer_size"] == 50
        assert not stats["is_finished"]


class TestSSEEventBuilder:
    def test_initialized_event(self):
        event = SSEEventBuilder.initialized("ready")
        assert event.event_type == StreamEventType.INITIALIZED
        assert event.data["message"] == "ready"

    def test_answer_chunk_event(self):
        event = SSEEventBuilder.answer_chunk("hello", conversation_id="c1", message_id="m1")
        assert event.event_type == StreamEventType.ANSWER_CHUNK
        assert event.data["answer"] == "hello"
        assert event.data["conversation_id"] == "c1"

    def test_answer_end_with_post_processor(self):
        def upper(text: str) -> str:
            return text.upper()

        event = SSEEventBuilder.answer_end("hello world", post_processors=[upper])
        assert event.data["final_answer"] == "HELLO WORLD"

    def test_error_event(self):
        event = SSEEventBuilder.error("something broke", "TestError")
        assert event.event_type == StreamEventType.ERROR
        assert event.data["error"] == "something broke"

    def test_thought_flow(self):
        start = SSEEventBuilder.thought_start(1)
        chunk = SSEEventBuilder.thought_chunk("thinking...", 1)
        end = SSEEventBuilder.thought_end("full thought", 1)
        assert start.iteration == 1
        assert chunk.data["content"] == "thinking..."
        assert end.data["full_thought"] == "full thought"

    def test_to_sse_message_format(self):
        event = SSEEventBuilder.completed("done")
        msg = event.to_sse_message()
        assert msg.startswith("data: ")
        assert msg.endswith("\n\n")
        assert '"completed"' in msg
