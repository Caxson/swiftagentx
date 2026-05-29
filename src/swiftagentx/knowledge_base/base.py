"""
KnowledgeBase abstract base class.
"""

from abc import ABC, abstractmethod

from .document import Document, SearchResult


class KnowledgeBase(ABC):
    """
    Abstract base class for knowledge base implementations.

    Subclass this and implement the abstract methods to integrate
    with any vector store (Weaviate, Elasticsearch, FAISS, etc.).
    """

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query text.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult sorted by score descending.
        """
        ...

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> int:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents to add.

        Returns:
            Number of documents successfully added.
        """
        ...

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: The document ID.

        Returns:
            True if the document was deleted, False if not found.
        """
        ...

    async def get_document(self, doc_id: str) -> Document | None:
        """Get a document by ID. Override for optimized implementations."""
        return None

    async def count(self) -> int:
        """Return the total number of documents. Override for optimized implementations."""
        return 0

    async def close(self) -> None:  # noqa: B027 — optional override, default no-op
        """Clean up resources. Override if your implementation needs cleanup."""
        pass
