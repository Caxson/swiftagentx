"""
AdminService — framework-agnostic management logic for a SwiftAgent agent.

Provides status queries, configuration management, cache operations,
and knowledge base management. Web framework adapters (Flask, FastAPI)
call into this service layer.

WARNING: Admin endpoints expose agent internals. Always add authentication
middleware (Flask before_request / FastAPI Depends) in production.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..core.agent import Agent

logger = logging.getLogger(__name__)

# Fields that may be updated at runtime via the admin API.
_MUTABLE_CONFIG_FIELDS = frozenset({
    "max_iterations", "max_retries", "enable_cache",
    "kb_cache_ttl", "code_cache_ttl", "max_input_length",
    "sse_heartbeat_interval", "sse_timeout", "log_level",
    "kb_exact_match_threshold", "debug",
    "max_cache_entries_per_level", "enable_request_tracing",
})


def _run_sync(coro):
    """Run a coroutine synchronously, safe in both sync and async contexts."""
    try:
        asyncio.get_running_loop()
        in_async = True
    except RuntimeError:
        in_async = False

    if in_async:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


class AdminService:
    """
    Stateless admin operations backed by an Agent instance.

    Usage::

        service = AdminService(agent)
        status = service.get_status()
    """

    def __init__(self, agent: "Agent") -> None:
        self.agent = agent
        self._start_time = time.time()

    # ---- Status ----

    def get_status(self) -> Dict[str, Any]:
        """Agent name, tool count, cache stats, uptime."""
        return {
            "name": self.agent.name,
            "tools": self.agent.tool_registry.list_tools(),
            "tool_count": self.agent.tool_registry.count(),
            "cache_stats": self.agent.cache.get_stats(),
            "has_knowledge_base": self.agent.knowledge_base is not None,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return JSON schema for every registered tool."""
        return [
            tool.get_schema()
            for tool in self.agent.tool_registry.get_all().values()
        ]

    def get_cache_stats(self) -> Dict[str, Any]:
        return self.agent.cache.get_stats()

    # ---- Configuration ----

    def get_config(self) -> Dict[str, Any]:
        """Return current config with sensitive values masked."""
        raw = self.agent.config.model_dump()
        return self._mask_secrets(raw)

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update config fields at runtime.

        Only whitelisted mutable fields are accepted; values are validated
        through Pydantic by re-constructing the config model.
        Returns the full (masked) config after the update.
        """
        from ..models.config import SwiftAgentConfig

        filtered = {k: v for k, v in updates.items() if k in _MUTABLE_CONFIG_FIELDS}
        if not filtered:
            return self.get_config()

        current = self.agent.config.model_dump()
        current.update(filtered)
        try:
            self.agent.config = SwiftAgentConfig(**current)
        except Exception as e:
            logger.warning(f"Config update rejected: {e}")
            return {"error": f"Invalid config: {e}"}

        return self.get_config()

    # ---- Cache management ----

    def clear_cache(self, level: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear cache entries.

        Args:
            level: One of "level_1", "level_2", "level_3", "scenario", or None (all).

        Returns:
            Confirmation dict.
        """
        if level is None:
            self.agent.cache.clear_all()
            return {"cleared": "all"}

        cache = self.agent.cache
        if level == "level_1":
            cache.level_1_cache.clear()
        elif level == "level_2":
            cache.level_2_cache.clear()
        elif level == "level_3":
            cache.level_3_cache.clear()
        elif level == "scenario":
            cache.clear_scenario_cache()
        else:
            return {"error": f"Unknown cache level: {level}"}

        return {"cleared": level}

    # ---- Knowledge base (sync wrappers) ----

    def kb_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Synchronous wrapper — safe in both sync and async contexts."""
        kb = self.agent.knowledge_base
        if kb is None:
            return []
        results = _run_sync(kb.search(query, top_k=top_k))
        return self._format_search_results(results)

    def kb_add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        kb = self.agent.knowledge_base
        if kb is None:
            return {"error": "No knowledge base configured"}
        from ..knowledge_base.document import Document
        try:
            docs = [Document(**d) for d in documents]
        except Exception as e:
            return {"error": f"Invalid document data: {e}"}
        added = _run_sync(kb.add_documents(docs))
        return {"added": added}

    def kb_delete_document(self, doc_id: str) -> Dict[str, Any]:
        kb = self.agent.knowledge_base
        if kb is None:
            return {"error": "No knowledge base configured"}
        deleted = _run_sync(kb.delete_document(doc_id))
        return {"deleted": deleted, "doc_id": doc_id}

    def kb_stats(self) -> Dict[str, Any]:
        kb = self.agent.knowledge_base
        if kb is None:
            return {"error": "No knowledge base configured"}
        count = _run_sync(kb.count())
        return {"document_count": count, "provider": type(kb).__name__}

    # ---- Knowledge base (async) ----

    async def kb_search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        kb = self.agent.knowledge_base
        if kb is None:
            return []
        results = await kb.search(query, top_k=top_k)
        return self._format_search_results(results)

    async def kb_add_documents_async(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        kb = self.agent.knowledge_base
        if kb is None:
            return {"error": "No knowledge base configured"}
        from ..knowledge_base.document import Document
        try:
            docs = [Document(**d) for d in documents]
        except Exception as e:
            return {"error": f"Invalid document data: {e}"}
        added = await kb.add_documents(docs)
        return {"added": added}

    async def kb_delete_document_async(self, doc_id: str) -> Dict[str, Any]:
        kb = self.agent.knowledge_base
        if kb is None:
            return {"error": "No knowledge base configured"}
        deleted = await kb.delete_document(doc_id)
        return {"deleted": deleted, "doc_id": doc_id}

    async def kb_stats_async(self) -> Dict[str, Any]:
        kb = self.agent.knowledge_base
        if kb is None:
            return {"error": "No knowledge base configured"}
        count = await kb.count()
        return {"document_count": count, "provider": type(kb).__name__}

    # ---- Helpers ----

    @staticmethod
    def _format_search_results(results) -> List[Dict[str, Any]]:
        return [
            {
                "doc_id": r.document.doc_id,
                "content": r.document.content,
                "score": r.score,
                "match_type": r.match_type,
                "metadata": r.document.metadata,
            }
            for r in results
        ]

    @staticmethod
    def _mask_secrets(data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask values that look like API keys or secrets."""
        masked = {}
        secret_pattern = re.compile(
            r"(key|secret|token|password|credential)", re.IGNORECASE
        )
        for key, value in data.items():
            if isinstance(value, str) and secret_pattern.search(key) and len(value) > 4:
                masked[key] = value[:4] + "***"
            elif isinstance(value, dict):
                masked[key] = AdminService._mask_secrets(value)
            elif isinstance(value, list):
                masked[key] = [
                    AdminService._mask_secrets(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                masked[key] = value
        return masked
