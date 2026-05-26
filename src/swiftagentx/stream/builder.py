"""
SSE event builder — convenience factory for creating stream events.
"""


from ..models.schema import StreamEvent, StreamEventType


class SSEEventBuilder:
    """Convenience class for creating various stream events."""

    @staticmethod
    def initialized(message: str = "Agent initialized", **kwargs: str) -> StreamEvent:
        return StreamEvent(event_type=StreamEventType.INITIALIZED, data={"message": message, **kwargs})

    @staticmethod
    def kb_lookup(query: str, status: str = "searching") -> StreamEvent:
        return StreamEvent(event_type=StreamEventType.KB_LOOKUP, data={"query": query, "status": status})

    @staticmethod
    def cache_hit(source: str, content: str | None = None) -> StreamEvent:
        return StreamEvent(event_type=StreamEventType.CACHE_HIT, data={"source": source, "content": content})

    @staticmethod
    def thought_start(iteration: int) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.THOUGHT_START, data={"iteration": iteration}, iteration=iteration,
        )

    @staticmethod
    def thought_chunk(content: str, iteration: int = 0) -> StreamEvent:
        return StreamEvent(event_type=StreamEventType.THOUGHT_CHUNK, data={"content": content}, iteration=iteration)

    @staticmethod
    def thought_end(full_thought: str, iteration: int = 0) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.THOUGHT_END, data={"full_thought": full_thought}, iteration=iteration,
        )

    @staticmethod
    def action_start(tool_name: str, iteration: int = 0) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.ACTION_START, data={"tool_name": tool_name}, iteration=iteration,
        )

    @staticmethod
    def action_chunk(content: str, iteration: int = 0) -> StreamEvent:
        return StreamEvent(event_type=StreamEventType.ACTION_CHUNK, data={"content": content}, iteration=iteration)

    @staticmethod
    def action_end(tool_name: str, tool_input: dict, iteration: int = 0) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.ACTION_END,
            data={"tool_name": tool_name, "tool_input": tool_input},
            iteration=iteration,
        )

    @staticmethod
    def observation(result: str, iteration: int = 0) -> StreamEvent:
        return StreamEvent(event_type=StreamEventType.OBSERVATION, data={"result": result}, iteration=iteration)

    @staticmethod
    def answer_start() -> StreamEvent:
        return StreamEvent(event_type=StreamEventType.ANSWER_START, data={"message": "Generating answer..."})

    @staticmethod
    def answer_chunk(
        content: str,
        conversation_id: str | None = None,
        message_id: str | None = None,
        task_id: str | None = None,
    ) -> StreamEvent:
        data: dict = {"answer": content}
        if conversation_id:
            data["conversation_id"] = conversation_id
        if message_id:
            data["message_id"] = message_id
        if task_id:
            data["task_id"] = task_id
        return StreamEvent(event_type=StreamEventType.ANSWER_CHUNK, data=data)

    @staticmethod
    def answer_end(
        final_answer: str,
        post_processors: list | None = None,
    ) -> StreamEvent:
        """Answer end event with optional post-processing hooks.

        Args:
            final_answer: The final answer text.
            post_processors: Optional list of callables that transform the answer text.
                Each callable takes a str and returns a str.
        """
        processed = final_answer
        if post_processors:
            for processor in post_processors:
                processed = processor(processed)
        return StreamEvent(event_type=StreamEventType.ANSWER_END, data={"final_answer": processed})

    @staticmethod
    def message_end(
        conversation_id: str,
        message_id: str,
        created_at: int,
        metadata: dict | None = None,
        task_id: str | None = None,
    ) -> StreamEvent:
        event_data: dict = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "created_at": created_at,
        }
        if task_id:
            event_data["task_id"] = task_id
        if metadata:
            event_data["metadata"] = metadata
        return StreamEvent(event_type=StreamEventType.MESSAGE_END, data=event_data)

    @staticmethod
    def completed(message: str = "Execution completed") -> StreamEvent:
        return StreamEvent(event_type=StreamEventType.COMPLETED, data={"message": message})

    @staticmethod
    def error(error_message: str, error_type: str = "UnknownError") -> StreamEvent:
        return StreamEvent(event_type=StreamEventType.ERROR, data={"error": error_message, "error_type": error_type})
