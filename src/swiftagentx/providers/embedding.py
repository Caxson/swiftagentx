"""
OpenAI-compatible embedding client.

Works with any service implementing the OpenAI ``/embeddings`` API —
OpenAI, Aliyun DashScope (``text-embedding-v4``), and other compatible
endpoints. Pair it with ``core.retrieval.EmbeddingRetriever`` to give the
scenario prefilter semantic matching:

    from swiftagentx import Agent, EmbeddingRetriever
    from swiftagentx.providers.embedding import OpenAICompatibleEmbeddingProvider

    embedder = OpenAICompatibleEmbeddingProvider(
        api_key=..., model="text-embedding-v4",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    agent = Agent(..., scenario_retriever=EmbeddingRetriever(embedder))
"""

import logging

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingProvider:
    """Implements the ``core.retrieval.Embedder`` protocol over HTTP.

    Args:
        api_key: API key
        model: Embedding model name (e.g. "text-embedding-v4")
        api_base: API base URL
        timeout_seconds: Per-request timeout
        batch_size: Max texts per request (DashScope caps batches at 10)
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str = "https://api.openai.com/v1",
        timeout_seconds: int = 30,
        batch_size: int = 10,
    ):
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleEmbeddingProvider requires the 'httpx' "
                "package. Install it with: pip install 'swiftagentx[openai]'"
            ) from exc
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.batch_size = max(1, batch_size)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, batching requests; order matches the input."""
        if not texts:
            return []
        import httpx

        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                response = await client.post(
                    f"{self.api_base}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "input": chunk},
                )
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda d: d["index"])
                if len(data) != len(chunk):
                    raise ValueError(
                        f"Embedding API returned {len(data)} vectors for "
                        f"{len(chunk)} inputs"
                    )
                vectors.extend(item["embedding"] for item in data)
        return vectors
