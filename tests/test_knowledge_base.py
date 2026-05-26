"""
Tests for the knowledge base module.

Covers: Document model, MemoryKnowledgeBase CRUD, TF-IDF search,
exact matching, KnowledgeBaseTool, KnowledgeBaseStage short-circuit.
"""

import pytest

from swiftagentx import (
    Agent,
    Document,
    DummyModelClient,
    KnowledgeBaseStage,
    KnowledgeBaseTool,
    MemoryKnowledgeBase,
    SearchResult,
)
from swiftagentx.core.pipeline import StageAction

# ---- fixtures ----

@pytest.fixture
def kb():
    return MemoryKnowledgeBase()


@pytest.fixture
def sample_docs():
    return [
        Document(doc_id="1", content="退货政策：7天无理由退换货", metadata={"category": "policy"}),
        Document(doc_id="2", content="会员积分可在商城兑换礼品", metadata={"category": "points"}),
        Document(doc_id="3", content="直播间每晚8点开播", metadata={"category": "live"}),
        Document(doc_id="4", content="商品价格以页面展示为准"),
        Document(doc_id="5", content="退货流程：联系客服申请退货，填写退货单"),
    ]


# ---- Document model ----

class TestDocument:
    def test_create_document(self):
        doc = Document(doc_id="test", content="hello world")
        assert doc.doc_id == "test"
        assert doc.content == "hello world"
        assert doc.metadata == {}

    def test_document_with_metadata(self):
        doc = Document(doc_id="x", content="text", metadata={"tag": "faq"})
        assert doc.metadata["tag"] == "faq"


class TestSearchResult:
    def test_create_search_result(self):
        doc = Document(doc_id="1", content="test")
        sr = SearchResult(document=doc, score=0.95, match_type="exact")
        assert sr.score == 0.95
        assert sr.match_type == "exact"


# ---- MemoryKnowledgeBase ----

class TestMemoryKnowledgeBase:
    @pytest.mark.asyncio
    async def test_add_documents(self, kb, sample_docs):
        added = await kb.add_documents(sample_docs)
        assert added == 5
        assert await kb.count() == 5

    @pytest.mark.asyncio
    async def test_get_document(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        doc = await kb.get_document("1")
        assert doc is not None
        assert doc.doc_id == "1"
        assert "退货" in doc.content

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, kb):
        doc = await kb.get_document("nonexistent")
        assert doc is None

    @pytest.mark.asyncio
    async def test_delete_document(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        deleted = await kb.delete_document("1")
        assert deleted is True
        assert await kb.count() == 4
        assert await kb.get_document("1") is None

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, kb):
        deleted = await kb.delete_document("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_exact_match(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        results = await kb.search("退货政策：7天无理由退换货")
        assert len(results) >= 1
        assert results[0].score == 1.0
        assert results[0].match_type == "exact"

    @pytest.mark.asyncio
    async def test_keyword_search(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        results = await kb.search("退货")
        assert len(results) >= 1
        # Should match docs containing 退货
        doc_ids = {r.document.doc_id for r in results}
        assert "1" in doc_ids or "5" in doc_ids

    @pytest.mark.asyncio
    async def test_search_no_results(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        results = await kb.search("quantum physics")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_kb(self, kb):
        results = await kb.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        results = await kb.search("退货", top_k=1)
        assert len(results) <= 1


# ---- KnowledgeBaseTool ----

class TestKnowledgeBaseTool:
    @pytest.mark.asyncio
    async def test_tool_search_exact(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        tool = KnowledgeBaseTool(kb=kb, exact_match_threshold=0.95)

        from swiftagentx.models.schema import SessionContext
        ctx = SessionContext(
            session_id="s1", user_id="u1",
            user_input="退货政策：7天无理由退换货",
        )
        result = await tool.execute(ctx, query="退货政策：7天无理由退换货")
        assert result.success
        assert result.output_type.value == "direct_output"
        assert "退货" in str(result.result)

    @pytest.mark.asyncio
    async def test_tool_search_keyword(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        tool = KnowledgeBaseTool(kb=kb, exact_match_threshold=0.95)

        from swiftagentx.models.schema import SessionContext
        ctx = SessionContext(
            session_id="s1", user_id="u1", user_input="退货",
        )
        result = await tool.execute(ctx, query="退货")
        assert result.success
        assert result.output_type.value == "llm_processed"

    @pytest.mark.asyncio
    async def test_tool_no_query(self, kb):
        tool = KnowledgeBaseTool(kb=kb)
        from swiftagentx.models.schema import SessionContext
        ctx = SessionContext(
            session_id="s1", user_id="u1", user_input="",
        )
        result = await tool.execute(ctx, query="")
        assert not result.success


# ---- KnowledgeBaseStage ----

class TestKnowledgeBaseStage:
    @pytest.mark.asyncio
    async def test_stage_short_circuit_on_exact(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        stage = KnowledgeBaseStage(kb=kb, threshold=0.95)

        context = {"user_input": "退货政策：7天无理由退换货", "session_id": "s1"}
        result = await stage.execute(context)
        assert result.action == StageAction.SHORT_CIRCUIT
        assert "退货" in result.answer

    @pytest.mark.asyncio
    async def test_stage_continue_on_keyword(self, kb, sample_docs):
        await kb.add_documents(sample_docs)
        stage = KnowledgeBaseStage(kb=kb, threshold=0.95)

        context = {"user_input": "退货", "session_id": "s1"}
        result = await stage.execute(context)
        assert result.action == StageAction.CONTINUE
        assert "kb_results" in context
        assert len(context["kb_results"]) > 0

    @pytest.mark.asyncio
    async def test_stage_continue_on_empty(self, kb):
        stage = KnowledgeBaseStage(kb=kb, threshold=0.95)
        context = {"user_input": "hello", "session_id": "s1"}
        result = await stage.execute(context)
        assert result.action == StageAction.CONTINUE


# ---- Agent integration ----

class TestAgentKBIntegration:
    @pytest.mark.asyncio
    async def test_set_knowledge_base(self):
        agent = Agent(
            model=DummyModelClient(api_key="test", model="dummy"),
        )
        kb = MemoryKnowledgeBase()
        await kb.add_documents([
            Document(doc_id="1", content="退货政策：7天无理由退换"),
        ])
        agent.set_knowledge_base(kb)

        assert agent.knowledge_base is kb
        assert "knowledge_base" in agent.tool_registry.list_tools()
