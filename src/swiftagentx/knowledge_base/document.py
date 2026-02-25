"""
Document data models for the knowledge base module.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A knowledge base document."""

    doc_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single search result from the knowledge base."""

    document: Document
    score: float = 0.0
    match_type: str = "keyword"  # "exact" | "semantic" | "keyword"
