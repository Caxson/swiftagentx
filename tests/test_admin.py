"""
Tests for the admin module.

Covers: AdminService status, tools, cache stats/clear, config get/update,
knowledge base management, and config masking.
"""

import pytest

from swiftagentx import (
    AdminService,
    Agent,
    DummyModelClient,
    MemoryKnowledgeBase,
)

# ---- fixtures ----

@pytest.fixture
def agent():
    return Agent(
        name="TestAgent",
        model=DummyModelClient(api_key="test-key", model="dummy"),
        max_iterations=3,
    )


@pytest.fixture
def service(agent):
    return AdminService(agent)


@pytest.fixture
def agent_with_kb(agent):
    kb = MemoryKnowledgeBase()
    agent.set_knowledge_base(kb)
    return agent


@pytest.fixture
def service_with_kb(agent_with_kb):
    return AdminService(agent_with_kb)


# ---- Status ----

class TestAdminStatus:
    def test_get_status(self, service):
        status = service.get_status()
        assert status["name"] == "TestAgent"
        assert isinstance(status["tools"], list)
        assert isinstance(status["cache_stats"], dict)
        assert status["has_knowledge_base"] is False
        assert status["uptime_seconds"] >= 0

    def test_get_status_with_kb(self, service_with_kb):
        status = service_with_kb.get_status()
        assert status["has_knowledge_base"] is True
        assert "knowledge_base" in status["tools"]

    def test_get_tools_empty(self, service):
        # "Empty" means no user-registered tools — `workspace_read` is a
        # built-in every Agent auto-registers for context offload (D3).
        tools = service.get_tools()
        assert isinstance(tools, list)
        assert [t["name"] for t in tools] == ["workspace_read"]

    def test_get_tools_with_kb(self, service_with_kb):
        tools = service_with_kb.get_tools()
        assert {t["name"] for t in tools} == {"workspace_read", "knowledge_base"}


# ---- Cache ----

class TestAdminCache:
    def test_get_cache_stats(self, service):
        stats = service.get_cache_stats()
        assert "total_hits" in stats
        assert stats["total_hits"] == 0

    def test_clear_cache_all(self, service):
        # Add something to cache first
        service.agent.cache.set_level_1("test", "value")
        result = service.clear_cache()
        assert result["cleared"] == "all"
        stats = service.get_cache_stats()
        assert stats["level_1_entries"] == 0

    def test_clear_cache_level_1(self, service):
        service.agent.cache.set_level_1("test", "value")
        result = service.clear_cache("level_1")
        assert result["cleared"] == "level_1"

    def test_clear_cache_unknown_level(self, service):
        result = service.clear_cache("level_99")
        assert "error" in result


# ---- Config ----

class TestAdminConfig:
    def test_get_config(self, service):
        config = service.get_config()
        assert config["name"] == "TestAgent"
        assert config["max_iterations"] == 3
        assert "kb_exact_match_threshold" in config

    def test_update_config(self, service):
        result = service.update_config({"max_iterations": 20})
        assert result["max_iterations"] == 20
        assert service.agent.config.max_iterations == 20

    def test_update_config_unknown_key(self, service):
        old_config = service.get_config()
        result = service.update_config({"nonexistent_key": "value"})
        # Unknown keys should be silently ignored
        assert result["name"] == old_config["name"]

    def test_update_config_invalid_type(self, service):
        result = service.update_config({"max_iterations": "not_a_number"})
        assert "error" in result
        # Original value should be preserved
        assert service.agent.config.max_iterations == 3

    def test_update_config_rejects_immutable_fields(self, service):
        result = service.update_config({"name": "HackedAgent"})
        # 'name' is not in _MUTABLE_CONFIG_FIELDS, should be ignored
        assert result["name"] == "TestAgent"

    def test_config_mask_secrets(self):
        data = {
            "name": "Agent",
            "api_key": "sk-1234567890abcdef",
            "password": "mysecret123",
            "debug": True,
        }
        masked = AdminService._mask_secrets(data)
        assert masked["name"] == "Agent"
        assert masked["api_key"] == "sk-1***"
        assert masked["password"] == "myse***"
        assert masked["debug"] is True

    def test_config_mask_secrets_handles_lists(self):
        data = {
            "providers": [
                {"name": "openai", "api_key": "sk-abcdefghij"},
                {"name": "local", "url": "http://localhost"},
            ],
        }
        masked = AdminService._mask_secrets(data)
        assert masked["providers"][0]["api_key"] == "sk-a***"
        assert masked["providers"][1]["url"] == "http://localhost"


# ---- KB Management ----

class TestAdminKB:
    @pytest.mark.asyncio
    async def test_kb_stats(self, service_with_kb):
        stats = await service_with_kb.kb_stats_async()
        assert stats["document_count"] == 0
        assert stats["provider"] == "MemoryKnowledgeBase"

    @pytest.mark.asyncio
    async def test_kb_add_and_search(self, service_with_kb):
        result = await service_with_kb.kb_add_documents_async([
            {"doc_id": "1", "content": "退货政策：7天无理由退换"},
            {"doc_id": "2", "content": "积分兑换规则"},
        ])
        assert result["added"] == 2

        results = await service_with_kb.kb_search_async("退货政策：7天无理由退换")
        assert len(results) >= 1
        assert results[0]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_kb_delete_document(self, service_with_kb):
        await service_with_kb.kb_add_documents_async([
            {"doc_id": "d1", "content": "test content"},
        ])
        result = await service_with_kb.kb_delete_document_async("d1")
        assert result["deleted"] is True

        result = await service_with_kb.kb_delete_document_async("d1")
        assert result["deleted"] is False

    @pytest.mark.asyncio
    async def test_kb_add_invalid_documents(self, service_with_kb):
        result = await service_with_kb.kb_add_documents_async([
            {"missing_doc_id_field": True},
        ])
        assert "error" in result

    @pytest.mark.asyncio
    async def test_kb_operations_without_kb(self, service):
        stats = await service.kb_stats_async()
        assert "error" in stats

        results = await service.kb_search_async("anything")
        assert results == []
