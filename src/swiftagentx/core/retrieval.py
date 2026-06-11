"""
Scenario retrieval pre-filter — keeps the classifier prompt O(K) as the
scenario pool grows.

Classification is a discrimination task: the light model's accuracy (and the
prompt size) degrades as more candidates are shown. Retrieval is a lookup
task that handles large pools trivially. This module ranks the scenario pool
against the user input so the router only puts the top-K most plausible
scenarios into the classification prompt — the pool can grow to hundreds of
scenarios while the classifier always sees a fixed-size candidate list.

The default :class:`LexicalRetriever` is zero-dependency and CJK-aware.
Plug in an embedding-based retriever by implementing the
:class:`ScenarioRetriever` protocol and passing it to ``IntentRouter``.
"""

import math
import re
from typing import Protocol, runtime_checkable

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
    ids in the result are tolerated and ignored by the router.
    """

    def rank(self, query: str, docs: dict[str, str]) -> list[str]: ...


class LexicalRetriever:
    """IDF-weighted token-overlap ranking. Zero dependencies, CJK-aware.

    Score(doc) = Σ idf(t) for t in (query ∩ doc), normalized by √|doc| so
    long documents don't win on bulk. IDF down-weights tokens shared across
    many scenarios (e.g. a "查询" that appears everywhere) so discriminative
    tokens decide the ranking. Ties keep registration order (stable sort).
    """

    def rank(self, query: str, docs: dict[str, str]) -> list[str]:
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
