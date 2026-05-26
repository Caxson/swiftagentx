"""
In-memory knowledge base implementation using TF-IDF + cosine similarity.

Zero external dependencies — uses only stdlib (math, collections, re).
Suitable for small-scale data (<10000 documents) and testing.
"""

import math
from collections import Counter

from .base import KnowledgeBase
from .document import Document, SearchResult


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: ASCII words + single CJK characters."""
    tokens: list[str] = []
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            tokens.append(char)
        elif char.isalnum():
            tokens.append(char.lower())
        else:
            tokens.append(' ')

    # Merge consecutive ASCII chars into words
    merged: list[str] = []
    buf: list[str] = []
    for t in tokens:
        if t == ' ':
            if buf:
                merged.append(''.join(buf))
                buf = []
        elif '\u4e00' <= t <= '\u9fff':
            if buf:
                merged.append(''.join(buf))
                buf = []
            merged.append(t)
        else:
            buf.append(t)
    if buf:
        merged.append(''.join(buf))

    return merged


class MemoryKnowledgeBase(KnowledgeBase):
    """
    In-memory knowledge base with TF-IDF search.

    Features:
    - Exact match (content equality → score=1.0, match_type="exact")
    - Keyword match (TF-IDF + cosine similarity → score=0~1, match_type="keyword")
    - Simple CJK support via single-character tokenization
    """

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        # Pre-computed TF-IDF data
        self._doc_tf: dict[str, dict[str, float]] = {}
        self._idf: dict[str, float] = {}
        self._dirty = True

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._documents:
            return []

        results: list[SearchResult] = []
        query_normalized = query.strip()

        # Phase 1: exact match
        for doc in self._documents.values():
            if doc.content.strip() == query_normalized:
                results.append(SearchResult(
                    document=doc, score=1.0, match_type="exact",
                ))

        if results:
            return results[:top_k]

        # Phase 2: TF-IDF keyword match
        if self._dirty:
            self._rebuild_index()

        query_tokens = _tokenize(query_normalized)
        if not query_tokens:
            return []

        query_tf = self._compute_tf(query_tokens)
        query_vec = {
            token: tf * self._idf.get(token, 0.0)
            for token, tf in query_tf.items()
        }

        for doc_id, doc_tfidf in self._doc_tf.items():
            doc_vec = {
                token: tf * self._idf.get(token, 0.0)
                for token, tf in doc_tfidf.items()
            }
            score = self._cosine_similarity(query_vec, doc_vec)
            if score > 0:
                results.append(SearchResult(
                    document=self._documents[doc_id],
                    score=round(score, 4),
                    match_type="keyword",
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def add_documents(self, documents: list[Document]) -> int:
        added = 0
        for doc in documents:
            self._documents[doc.doc_id] = doc
            added += 1
        if added > 0:
            self._dirty = True
        return added

    async def delete_document(self, doc_id: str) -> bool:
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._doc_tf.pop(doc_id, None)
            self._dirty = True
            return True
        return False

    async def get_document(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    async def count(self) -> int:
        return len(self._documents)

    # --- Internal ---

    def _rebuild_index(self) -> None:
        """Rebuild TF-IDF index for all documents."""
        self._doc_tf.clear()
        doc_count = len(self._documents)
        if doc_count == 0:
            self._idf.clear()
            self._dirty = False
            return

        # Compute TF for each document
        df: Counter = Counter()
        for doc_id, doc in self._documents.items():
            tokens = _tokenize(doc.content)
            tf = self._compute_tf(tokens)
            self._doc_tf[doc_id] = tf
            for token in tf:
                df[token] += 1

        # Compute IDF
        self._idf = {
            token: math.log((doc_count + 1) / (count + 1)) + 1
            for token, count in df.items()
        }
        self._dirty = False

    @staticmethod
    def _compute_tf(tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        return {token: count / total for token, count in counts.items()}

    @staticmethod
    def _cosine_similarity(
        vec_a: dict[str, float], vec_b: dict[str, float]
    ) -> float:
        common_keys = set(vec_a) & set(vec_b)
        if not common_keys:
            return 0.0

        dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
