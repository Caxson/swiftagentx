"""
In-memory storage backend — zero dependencies, works out of the box.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any

from .base import StorageBackend


class MemoryStorage(StorageBackend):
    """
    In-memory storage backend.

    Stores everything in Python dicts. Suitable for development and testing.
    Data is lost when the process exits.
    """

    def __init__(self, max_messages_per_conversation: int = 100) -> None:
        self._conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._request_logs: list[dict[str, Any]] = []
        self._max_messages = max_messages_per_conversation

    async def save_message(
        self, conversation_id: str, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> bool:
        message = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }
        messages = self._conversations[conversation_id]
        messages.append(message)
        if len(messages) > self._max_messages:
            self._conversations[conversation_id] = messages[-self._max_messages:]
        return True

    async def get_history(self, conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
        messages = self._conversations.get(conversation_id, [])
        return messages[-limit:] if len(messages) > limit else messages

    async def save_request_log(self, request_data: dict[str, Any]) -> bool:
        request_data.setdefault("logged_at", datetime.now().isoformat())
        self._request_logs.append(request_data)
        return True

    def get_all_conversations(self) -> dict[str, list[dict[str, Any]]]:
        return dict(self._conversations)

    def get_request_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._request_logs[-limit:]

    def clear(self) -> None:
        self._conversations.clear()
        self._request_logs.clear()
