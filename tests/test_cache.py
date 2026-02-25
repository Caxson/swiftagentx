"""
Tests for the three-level cache system.
"""

import pytest
from swiftagentx.core.cache import CacheManager


class TestCacheManager:
    def test_level_1_set_and_query(self):
        cache = CacheManager()
        cache.set_level_1("test query", "cached answer", ttl_seconds=3600)
        hit, value, source = cache.query("test query", "user1", "sess1")
        assert hit
        assert value == "cached answer"
        assert source == "level_1_kb"

    def test_level_2_set_and_query(self):
        cache = CacheManager()
        cache.set_level_2("query", "user1", "tool result", ttl_seconds=300)
        hit, value, source = cache.query("query", "user1", "sess1")
        assert hit
        assert value == "tool result"
        assert source == "level_2_code"

    def test_level_3_set_and_query(self):
        cache = CacheManager()
        cache.set_level_3("sess1", "var_key", "var_value")
        hit, value, source = cache.query("var_key", "user1", "sess1")
        assert hit
        assert value == "var_value"
        assert source == "level_3_dynamic"

    def test_cache_miss(self):
        cache = CacheManager()
        hit, value, source = cache.query("unknown", "user1", "sess1")
        assert not hit
        assert value is None

    def test_scenario_cache(self):
        cache = CacheManager()
        cache.set_scenario_cache("weather", "beijing_2026-01-01", "sunny", ttl_seconds=3600)
        result = cache.get_scenario_cache("weather", "beijing_2026-01-01")
        assert result == "sunny"

    def test_scenario_cache_miss(self):
        cache = CacheManager()
        result = cache.get_scenario_cache("weather", "nonexistent")
        assert result is None

    def test_clear_all(self):
        cache = CacheManager()
        cache.set_level_1("q1", "v1")
        cache.set_level_2("q2", "u1", "v2")
        cache.set_level_3("s1", "k3", "v3")
        cache.set_scenario_cache("sc", "key", "val")
        cache.clear_all()
        stats = cache.get_stats()
        assert stats["level_1_entries"] == 0
        assert stats["level_2_entries"] == 0
        assert stats["level_3_sessions"] == 0
        assert stats["scenario_entries"] == 0

    def test_stats_tracking(self):
        cache = CacheManager()
        cache.set_level_1("q", "v")
        cache.query("q", "u", "s")
        cache.query("q", "u", "s")
        stats = cache.get_stats()
        assert stats["level_1_hits"] == 2
        assert stats["total_hits"] == 2

    def test_platform_isolation(self):
        cache = CacheManager()
        cache.set_level_2("points", "user1", "100", platform="android")
        cache.set_level_2("points", "user1", "200", platform="ios")
        hit1, val1, _ = cache.query("points", "user1", "s1", platform="android")
        hit2, val2, _ = cache.query("points", "user1", "s1", platform="ios")
        assert hit1 and val1 == "100"
        assert hit2 and val2 == "200"

    def test_uses_sha256_not_md5(self):
        import hashlib
        cache = CacheManager()
        expected = hashlib.sha256("test".encode()).hexdigest()
        actual = cache._hash_key("test")
        assert actual == expected
        assert len(actual) == 64  # SHA-256 hex digest length

    def test_eviction_on_max_entries(self):
        cache = CacheManager(max_entries_per_level=3)
        for i in range(5):
            cache.set_level_1(f"query_{i}", f"value_{i}")
        assert len(cache.level_1_cache) == 3

    def test_auto_cleanup_on_access(self):
        cache = CacheManager()
        cache.set_level_1("q", "v", ttl_seconds=0)  # immediately expired
        # Access 100 times to trigger cleanup
        for _ in range(100):
            cache.query("miss", "u", "s")
        assert len(cache.level_1_cache) == 0
