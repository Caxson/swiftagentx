"""
Storage backend abstract interface.
"""

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """
    Abstract storage backend.

    Implement this to persist conversations, request logs, and agent state.
    """

    @abstractmethod
    async def save_message(
        self, conversation_id: str, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> bool:
        ...

    @abstractmethod
    async def get_history(self, conversation_id: str, limit: int = 10) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def save_request_log(self, request_data: dict[str, Any]) -> bool:
        ...

    async def close(self) -> None:  # noqa: B027 — optional override, default no-op
        """Clean up resources."""
        pass
