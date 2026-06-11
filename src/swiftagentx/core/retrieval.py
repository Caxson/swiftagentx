"""
Scenario retrieval pre-filter — keeps the classifier prompt O(K) as the
scenario pool grows.

Classification is a discrimination task: the light model's accuracy (and the
prompt size) degrades as more candidates are shown. Retrieval is a lookup
task that handles large pools trivially. This module ranks the scenario pool
against the user input so the router only puts the top-K most plausible
scenarios into the classification prompt — the pool can grow to hundreds of
scenarios while the classifier always sees a fixed-size candidate list.

Two built-in retrievers:

- :class:`LexicalRetriever` (default) — zero-dependency, CJK-aware token
  overlap. Zero added latency, but only matches shared surface tokens.
- :class:`EmbeddingRetriever` — semantic ranking via any :class:`Embedder`
  (e.g. ``providers.embedding.OpenAICompatibleEmbeddingProvider``).
  Catches phrasings with no lexical overlap ("出门要带伞吗" → weather).
  Doc vectors are embedded once and cached; each request costs one query
  embedding. Falls back to lexical when the embedding call fails.
"""

import logging
import math
import re
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_LATIN_TOKEN = re.compile(r"[a-z0-9]+")
# Han + Hiragana/Katakana + Hangul runs (punctuation/whitespace break a run,
# so bigrams never span phrase boundaries).
_CJK_RUN = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]+")


def tokenize(text: str) -> list[str]:
    """Split text into matchable tokens: Latin/digit words + CJK bigrams.

    CJK has no whitespace word boundaries, so character bigrams within each
    contiguous CJK run are the standard zero-dependency unit ("天气预报" →
    "天气", "气预", "预报"). A single-character run is kept as-is so it
    doesn't vanish.
    """
    lowered = text.lower()
    tokens = _LATIN_TOKEN.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@runtime_checkable
class ScenarioRetriever(Protocol):
    """Ranks candidate documents against a query.

    ``rank`` must return doc ids sorted by relevance (best first), ideally a
    permutation of ``docs.keys()`` — the caller truncates to top-K. Unknown
    ids in the result are tolerated and ignored by the router. Async so
    implementations may call out to an embedding service.
    """

    async def rank(self, query: str, docs: dict[str, str]) -> list[str]: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into vectors. One call may embed a batch."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class LexicalRetriever:
    """IDF-weighted token-overlap ranking. Zero dependencies, CJK-aware.

    Score(doc) = Σ idf(t) for t in (query ∩ doc), normalized by √|doc| so
    long documents don't win on bulk. IDF down-weights tokens shared across
    many scenarios (e.g. a "查询" that appears everywhere) so discriminative
    tokens decide the ranking. Ties keep registration order (stable sort).
    """

    async def rank(self, query: str, docs: dict[str, str]) -> list[str]:
        if not docs:
            return []
        query_tokens = set(tokenize(query))
        doc_tokens = {doc_id: set(tokenize(text)) for doc_id, text in docs.items()}

        n_docs = len(docs)
        df: dict[str, int] = {}
        for tokens in doc_tokens.values():
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1

        def score(tokens: set[str]) -> float:
            overlap = query_tokens & tokens
            if not overlap:
                return 0.0
            weight = sum(math.log(1.0 + n_docs / df[tok]) for tok in overlap)
            return weight / math.sqrt(len(tokens))

        scores = {doc_id: score(tokens) for doc_id, tokens in doc_tokens.items()}
        return sorted(docs.keys(), key=lambda doc_id: -scores[doc_id])


class EmbeddingRetriever:
    """Semantic ranking via an :class:`Embedder`, lexical fallback on error.

    Doc vectors are cached by document text — scenario docs are stable
    across requests, so after the first classification only the query
    itself is embedded (one call per request). Any embedding failure
    degrades to the fallback retriever instead of failing classification.
    """

    # Hard cap on cached doc vectors; reaching it means doc texts are being
    # regenerated per call (a bug) — reset instead of growing unbounded.
    MAX_CACHED_DOCS = 4096

    def __init__(
        self,
        embedder: Embedder,
        fallback: ScenarioRetriever | None = None,
    ):
        self._embedder = embedder
        self._fallback: ScenarioRetriever = (
            fallback if fallback is not None else LexicalRetriever()
        )
        self._doc_vectors: dict[str, list[float]] = {}

    async def rank(self, query: str, docs: dict[str, str]) -> list[str]:
        if not docs:
            return []
        try:
            await self._ensure_doc_vectors(docs.values())
            query_vector = (await self._embedder.embed([query]))[0]
            scores = {
                doc_id: _cosine(query_vector, self._doc_vectors[text])
                for doc_id, text in docs.items()
            }
            return sorted(docs.keys(), key=lambda doc_id: -scores[doc_id])
        except Exception:
            logger.error(
                "Embedding retrieval failed; falling back to lexical",
                exc_info=True,
            )
            return await self._fallback.rank(query, docs)

    async def _ensure_doc_vectors(self, texts: Iterable[str]) -> None:
        missing = list(dict.fromkeys(
            text for text in texts if text not in self._doc_vectors
        ))
        if not missing:
            return
        if len(self._doc_vectors) + len(missing) > self.MAX_CACHED_DOCS:
            logger.warning("EmbeddingRetriever doc cache overflow; resetting")
            self._doc_vectors.clear()
        vectors = await self._embedder.embed(missing)
        if len(vectors) != len(missing):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(missing)} texts"
            )
        for text, vector in zip(missing, vectors, strict=True):
            self._doc_vectors[text] = vector


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / norm if norm else 0.0
