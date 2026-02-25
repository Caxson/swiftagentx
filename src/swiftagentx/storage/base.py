"""
Storage backend abstract interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StorageBackend(ABC):
    """
    Abstract storage backend.

    Implement this to persist conversations, request logs, and agent state.
    """

    @abstractmethod
    async def save_message(
        self, conversation_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        ...

    @abstractmethod
    async def get_history(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def save_request_log(self, request_data: Dict[str, Any]) -> bool:
        ...

    async def close(self) -> None:
        """Clean up resources."""
        pass
