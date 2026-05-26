"""
Session memory management — manages conversation history and execution steps.
"""

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Conversation message."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.now()


class SessionMemory:
    """
    Session memory manager.

    Manages conversation history with automatic cleanup based on message count and age.
    """

    def __init__(self, max_messages: int = 30, max_age_minutes: int = 60):
        self.max_messages = max_messages
        self.max_age_minutes = max_age_minutes
        self.messages: list[Message] = []

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> Message:
        message = Message(role=role, content=content, timestamp=datetime.now(), metadata=metadata or {})
        self.messages.append(message)
        self._cleanup()
        return message

    def get_messages(self, limit: int | None = None) -> list[Message]:
        self._cleanup()
        if limit is None:
            return self.messages
        return self.messages[-limit:]

    def get_conversation_text(self, limit: int | None = None) -> str:
        """Get conversation as formatted text (for prompt construction)."""
        messages = self.get_messages(limit)
        return "\n".join(f"{msg.role.upper()}: {msg.content}" for msg in messages)

    def get_messages_dicts(self, limit: int | None = None) -> list[dict[str, str]]:
        """Get messages as list of dicts (for LLM API calls)."""
        messages = self.get_messages(limit)
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def get_conversation_for_reply(self, rounds: int = 5) -> list[dict[str, str]]:
        """Get recent conversation rounds for reply generation (user + assistant only)."""
        self._cleanup()
        conversation = [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
            if msg.role in ("user", "assistant")
        ]
        limit = rounds * 2
        return conversation[-limit:] if len(conversation) > limit else conversation

    def get_context_for_classification(self) -> str:
        """Get minimal context for intent classification (empty by default)."""
        return ""

    def get_context_for_execution(self, user_id: str, session_id: str) -> dict[str, Any]:
        """Get context needed for tool execution."""
        self._cleanup()
        return {
            "user_id": user_id,
            "session_id": session_id,
            "message_count": len(self.messages),
        }

    def clear(self) -> None:
        self.messages.clear()

    def _cleanup(self) -> None:
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=self.max_age_minutes)
        self.messages = [
            msg for msg in self.messages
            if msg.timestamp is None or msg.timestamp >= cutoff_time
        ]
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_stats(self) -> dict[str, Any]:
        self._cleanup()
        return {
            "total_messages": len(self.messages),
            "user_messages": sum(1 for msg in self.messages if msg.role == "user"),
            "assistant_messages": sum(1 for msg in self.messages if msg.role == "assistant"),
            "max_messages": self.max_messages,
            "oldest_message": self.messages[0].timestamp if self.messages else None,
            "newest_message": self.messages[-1].timestamp if self.messages else None,
        }

    def export(self) -> dict[str, Any]:
        self._cleanup()
        return {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "metadata": msg.metadata,
                }
                for msg in self.messages
            ],
            "stats": self.get_stats(),
        }

    def __len__(self) -> int:
        self._cleanup()
        return len(self.messages)

    def __repr__(self) -> str:
        return f"<SessionMemory: {len(self)} messages>"
