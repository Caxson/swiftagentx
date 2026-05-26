"""
Three-level cache system.

Level 1: Knowledge base cache (exact match, long TTL)
Level 2: Code/tool result cache (per-user, configurable TTL)
Level 3: Dynamic/session cache (session variables, no expiry)
Scenario cache: Pre-defined scenario toolchain cache
"""

import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class CacheEntry:
    """Single cache entry with TTL."""

    def __init__(
        self,
        key: str,
        value: Any,
        cache_type: str,
        ttl_seconds: int = 300,
        metadata: dict[str, Any] | None = None,
    ):
        self.key = key
        self.value = value
        self.cache_type = cache_type
        self.created_at = datetime.now()
        self.ttl_seconds = ttl_seconds
        self.metadata = metadata or {}
        self.hits = 0

    @property
    def expired_at(self) -> datetime:
        return self.created_at + timedelta(seconds=self.ttl_seconds)

    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expired_at

    def hit(self) -> None:
        self.hits += 1


class CacheManager:
    """
    Three-level cache manager.

    - Level 1: Knowledge base (similarity > 0.95)
    - Level 2: Tool result cache (per user)
    - Level 3: Session variables (no expiry)
    - Scenario: Pre-defined scenario toolchain cache
    """

    def __init__(self, max_entries_per_level: int = 10000) -> None:
        self.max_entries_per_level = max_entries_per_level
        self.level_1_cache: dict[str, CacheEntry] = {}
        self.level_2_cache: dict[str, dict[str, CacheEntry]] = {}
        self.level_3_cache: dict[str, dict[str, Any]] = {}
        self.scenario_cache: dict[str, CacheEntry] = {}
        self.stats = {"level_1_hits": 0, "level_2_hits": 0, "level_3_hits": 0, "scenario_hits": 0}
        self._access_count = 0

    def query(
        self, query: str, user_id: str, session_id: str, platform: str = "default"
    ) -> tuple[bool, Any | None, str]:
        """
        Query all cache levels in order.

        Returns:
            (hit, value, source) tuple
        """
        # Periodic cleanup (every 100 accesses)
        self._access_count += 1
        if self._access_count % 100 == 0:
            self.clear_expired()

        # Level 1: KB cache
        hit, value = self._query_level_1(query)
        if hit:
            self.stats["level_1_hits"] += 1
            logger.info(f"Cache hit (Level 1 - KB): {query[:50]}")
            return True, value, "level_1_kb"

        # Level 2: Tool result cache
        hit, value = self._query_level_2(query, user_id, platform)
        if hit:
            self.stats["level_2_hits"] += 1
            logger.info(f"Cache hit (Level 2 - Code): {query[:50]}")
            return True, value, "level_2_code"

        # Level 3: Session variable cache
        hit, value = self._query_level_3(query, session_id)
        if hit:
            self.stats["level_3_hits"] += 1
            logger.info(f"Cache hit (Level 3 - Dynamic): {query[:50]}")
            return True, value, "level_3_dynamic"

        return False, None, ""

    def _query_level_1(self, query: str) -> tuple[bool, Any | None]:
        query_hash = self._hash_key(query)
        if query_hash in self.level_1_cache:
            entry = self.level_1_cache[query_hash]
            if not entry.is_expired:
                entry.hit()
                return True, entry.value
            else:
                del self.level_1_cache[query_hash]
        return False, None

    def _query_level_2(self, query: str, user_id: str, platform: str = "default") -> tuple[bool, Any | None]:
        if user_id not in self.level_2_cache:
            return False, None
        query_hash = self._hash_key(f"{user_id}:{platform}:{query}")
        if query_hash in self.level_2_cache[user_id]:
            entry = self.level_2_cache[user_id][query_hash]
            if not entry.is_expired:
                entry.hit()
                return True, entry.value
            else:
                del self.level_2_cache[user_id][query_hash]
        return False, None

    def _query_level_3(self, query: str, session_id: str) -> tuple[bool, Any | None]:
        if session_id not in self.level_3_cache:
            return False, None
        if query in self.level_3_cache[session_id]:
            return True, self.level_3_cache[session_id][query]
        return False, None

    def _evict_oldest(self, cache: dict[str, CacheEntry]) -> None:
        """Evict oldest entries when cache exceeds max size."""
        if self.max_entries_per_level <= 0 or len(cache) <= self.max_entries_per_level:
            return
        # Sort by creation time, remove oldest entries
        sorted_keys = sorted(cache.keys(), key=lambda k: cache[k].created_at)
        excess = len(cache) - self.max_entries_per_level
        for key in sorted_keys[:excess]:
            del cache[key]

    def set_level_1(self, query: str, value: Any, ttl_seconds: int = 3600) -> None:
        query_hash = self._hash_key(query)
        self.level_1_cache[query_hash] = CacheEntry(
            key=query_hash, value=value, cache_type="level_1_kb",
            ttl_seconds=ttl_seconds, metadata={"query": query},
        )
        self._evict_oldest(self.level_1_cache)

    def set_level_2(
        self, query: str, user_id: str, value: Any, ttl_seconds: int = 300, platform: str = "default"
    ) -> None:
        if user_id not in self.level_2_cache:
            self.level_2_cache[user_id] = {}
        query_hash = self._hash_key(f"{user_id}:{platform}:{query}")
        self.level_2_cache[user_id][query_hash] = CacheEntry(
            key=query_hash, value=value, cache_type="level_2_code",
            ttl_seconds=ttl_seconds, metadata={"query": query, "user_id": user_id, "platform": platform},
        )
        self._evict_oldest(self.level_2_cache[user_id])

    def set_level_3(self, session_id: str, key: str, value: Any) -> None:
        if session_id not in self.level_3_cache:
            self.level_3_cache[session_id] = {}
        self.level_3_cache[session_id][key] = value

    def get_session_cache(self, session_id: str) -> dict[str, Any]:
        return self.level_3_cache.get(session_id, {}).copy()

    def clear_session_cache(self, session_id: str) -> None:
        self.level_3_cache.pop(session_id, None)

    def clear_user_cache(self, user_id: str) -> None:
        self.level_2_cache.pop(user_id, None)

    # --- Scenario cache ---

    def get_scenario_cache(self, scenario: str, cache_key: str) -> Any | None:
        full_key = f"scenario:{scenario}:{cache_key}"
        if full_key in self.scenario_cache:
            entry = self.scenario_cache[full_key]
            if not entry.is_expired:
                entry.hit()
                self.stats["scenario_hits"] += 1
                logger.info(f"Scenario cache hit: {scenario} -> {cache_key}")
                return entry.value
            else:
                del self.scenario_cache[full_key]
        return None

    def set_scenario_cache(self, scenario: str, cache_key: str, value: Any, ttl_seconds: int = 3600) -> None:
        full_key = f"scenario:{scenario}:{cache_key}"
        self.scenario_cache[full_key] = CacheEntry(
            key=full_key, value=value, cache_type=f"scenario_{scenario}",
            ttl_seconds=ttl_seconds, metadata={"scenario": scenario, "cache_key": cache_key},
        )
        self._evict_oldest(self.scenario_cache)

    def clear_scenario_cache(self, scenario: str | None = None) -> None:
        if scenario:
            prefix = f"scenario:{scenario}:"
            keys_to_delete = [k for k in self.scenario_cache if k.startswith(prefix)]
            for key in keys_to_delete:
                del self.scenario_cache[key]
        else:
            self.scenario_cache.clear()

    def clear_expired(self) -> None:
        expired_l1 = [k for k, e in self.level_1_cache.items() if e.is_expired]
        for key in expired_l1:
            del self.level_1_cache[key]

        for user_id in list(self.level_2_cache.keys()):
            expired_l2 = [k for k, e in self.level_2_cache[user_id].items() if e.is_expired]
            for key in expired_l2:
                del self.level_2_cache[user_id][key]
            if not self.level_2_cache[user_id]:
                del self.level_2_cache[user_id]

        expired_sc = [k for k, e in self.scenario_cache.items() if e.is_expired]
        for key in expired_sc:
            del self.scenario_cache[key]

    def get_stats(self) -> dict[str, Any]:
        total_hits = sum(self.stats.values())
        return {
            **self.stats,
            "total_hits": total_hits,
            "level_1_entries": len(self.level_1_cache),
            "level_2_entries": sum(len(v) for v in self.level_2_cache.values()),
            "level_3_sessions": len(self.level_3_cache),
            "scenario_entries": len(self.scenario_cache),
        }

    def clear_all(self) -> None:
        self.level_1_cache.clear()
        self.level_2_cache.clear()
        self.level_3_cache.clear()
        self.scenario_cache.clear()
        self.stats = {"level_1_hits": 0, "level_2_hits": 0, "level_3_hits": 0, "scenario_hits": 0}

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()
