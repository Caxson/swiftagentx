"""Tests for the scenario retrieval pre-filter (core/retrieval.py + router integration)."""

import pytest

from swiftagentx.core.model_client import DummyModelClient, ModelResponse
from swiftagentx.core.retrieval import LexicalRetriever, tokenize
from swiftagentx.core.router import IntentLevel, IntentRouter

# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_latin_words_lowercased(self):
        assert tokenize("Send Email to Bob") == ["send", "email", "to", "bob"]

    def test_cjk_bigrams(self):
        assert tokenize("查天气") == ["查天", "天气"]

    def test_cjk_single_char_run(self):
        # A 1-char CJK run must still produce a token, not vanish.
        assert "查" in tokenize("查 IP")

    def test_cjk_runs_do_not_bigram_across_boundaries(self):
        # "查天气，发邮件" — no spurious "气发" bigram across the comma.
        tokens = tokenize("查天气，发邮件")
        assert "天气" in tokens and "邮件" in tokens
        assert "气发" not in tokens

    def test_mixed_cjk_latin(self):
        tokens = tokenize("用Python查询天气API")
        assert "python" in tokens
        assert "api" in tokens
        assert "天气" in tokens

    def test_empty_and_punctuation_only(self):
        assert tokenize("") == []
        assert tokenize("！？。，") == []


# ---------------------------------------------------------------------------
# LexicalRetriever
# ---------------------------------------------------------------------------

DOCS = {
    "weather": "weather_query 天气查询 查询城市天气 天气 气温 下雨",
    "email": "send_email 发送邮件 发邮件给某人 邮件 收件人",
    "crm": "crm_lookup 客户查询 查客户资料 客户 CRM",
    "calc": "calculator 计算器 算术计算 计算",
}


class TestLexicalRetriever:
    @pytest.mark.asyncio
    async def test_relevant_doc_ranked_first_cjk(self):
        ranked = await LexicalRetriever().rank("北京今天天气怎么样", DOCS)
        assert ranked[0] == "weather"

    @pytest.mark.asyncio
    async def test_relevant_doc_ranked_first_latin(self):
        ranked = await LexicalRetriever().rank("please send an email to alice", DOCS)
        assert ranked[0] == "email"

    @pytest.mark.asyncio
    async def test_returns_permutation_of_all_ids(self):
        ranked = await LexicalRetriever().rank("天气", DOCS)
        assert sorted(ranked) == sorted(DOCS.keys())

    @pytest.mark.asyncio
    async def test_zero_overlap_keeps_insertion_order(self):
        ranked = await LexicalRetriever().rank("完全无关的输入xyz", DOCS)
        assert ranked == list(DOCS.keys())

    @pytest.mark.asyncio
    async def test_empty_docs(self):
        assert await LexicalRetriever().rank("天气", {}) == []

    @pytest.mark.asyncio
    async def test_common_token_does_not_dominate(self):
        # "查询" appears in several docs (low IDF); the discriminative
        # token "客户" must decide the winner.
        ranked = await LexicalRetriever().rank("查询客户信息", DOCS)
        assert ranked[0] == "crm"


# ---------------------------------------------------------------------------
# EmbeddingRetriever
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    """Maps known texts to fixed vectors; counts embed() calls."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [self.table[t] for t in texts]


class _ExplodingEmbedder:
    async def embed(self, texts):
        raise RuntimeError("embedding service down")


class TestEmbeddingRetriever:
    @pytest.mark.asyncio
    async def test_ranks_by_cosine_similarity(self):
        from swiftagentx.core.retrieval import EmbeddingRetriever

        docs = {"weather": "天气查询", "email": "发送邮件"}
        embedder = _FakeEmbedder({
            "天气查询": [1.0, 0.0],
            "发送邮件": [0.0, 1.0],
            "出门要带伞吗": [0.9, 0.1],   # semantically weather, zero lexical overlap
        })
        ranked = await EmbeddingRetriever(embedder).rank("出门要带伞吗", docs)
        assert ranked[0] == "weather"

    @pytest.mark.asyncio
    async def test_doc_vectors_cached_across_calls(self):
        from swiftagentx.core.retrieval import EmbeddingRetriever

        docs = {"weather": "天气查询"}
        embedder = _FakeEmbedder({"天气查询": [1.0, 0.0], "q1": [1.0, 0.0], "q2": [0.5, 0.5]})
        retriever = EmbeddingRetriever(embedder)
        await retriever.rank("q1", docs)
        await retriever.rank("q2", docs)
        # call 1: docs, call 2: q1, call 3: q2 — docs NOT re-embedded.
        assert embedder.calls == 3

    @pytest.mark.asyncio
    async def test_falls_back_to_lexical_on_embedding_failure(self):
        from swiftagentx.core.retrieval import EmbeddingRetriever

        ranked = await EmbeddingRetriever(_ExplodingEmbedder()).rank("北京天气", DOCS)
        assert ranked[0] == "weather"  # lexical fallback still ranks correctly


# ---------------------------------------------------------------------------
# Router pre-filter integration
# ---------------------------------------------------------------------------

class PromptCapturingModel(DummyModelClient):
    """Returns a fixed classification and records the prompt it was given."""

    def __init__(self, reply: str = "level=3"):
        super().__init__(api_key="test", model="capture")
        self.reply = reply
        self.last_prompt: str = ""

    async def chat(self, messages, **kwargs) -> ModelResponse:
        self.last_prompt = messages[-1]["content"]
        return ModelResponse(content=self.reply, model="capture")


def _make_scenarios(n: int) -> dict[str, dict]:
    """n filler scenarios plus one real weather scenario."""
    pool = {
        f"filler_{i}": {
            "name": f"填充场景{i}",
            "description": f"专用业务流程编号{i}",
            "triggers": [f"触发词{i}"],
            "slots": [],
        }
        for i in range(n)
    }
    pool["weather_query"] = {
        "name": "天气查询",
        "description": "查询指定城市的天气",
        "triggers": ["天气", "气温", "下雨"],
        "slots": ["city"],
    }
    return pool


class TestRouterPrefilter:
    @pytest.mark.asyncio
    async def test_large_pool_prompt_shows_only_top_k(self):
        router = IntentRouter(scenarios=_make_scenarios(30), prefilter_top_k=8)
        model = PromptCapturingModel("level=2 scenario=weather_query slots={\"city\": \"北京\"}")
        result = await router.classify("北京今天天气怎么样", model=model)

        # The matching scenario survived the filter and classified correctly.
        assert result.level == IntentLevel.SCENARIO
        assert result.scenario == "weather_query"
        assert "weather_query" in model.last_prompt
        # Prompt holds at most K candidates, not the whole pool of 31.
        shown = model.last_prompt.count("(填充场景") + model.last_prompt.count("(天气查询")
        assert shown <= 8
        assert result.metadata["prefilter"]["pool"] == 31

    @pytest.mark.asyncio
    async def test_small_pool_is_noop(self):
        scenarios = {
            "weather_query": {"name": "天气查询", "slots": ["city"]},
            "send_email": {"name": "发送邮件", "slots": ["to"]},
        }
        router = IntentRouter(scenarios=scenarios, prefilter_top_k=8)
        model = PromptCapturingModel()
        result = await router.classify("随便聊聊", model=model)

        # Both scenarios stay in the prompt; no prefilter metadata recorded.
        assert "weather_query" in model.last_prompt
        assert "send_email" in model.last_prompt
        assert "prefilter" not in result.metadata

    @pytest.mark.asyncio
    async def test_broken_retriever_falls_back_to_full_pool(self):
        class BrokenRetriever:
            async def rank(self, query, docs):
                raise RuntimeError("boom")

        router = IntentRouter(
            scenarios=_make_scenarios(30),
            retriever=BrokenRetriever(),
            prefilter_top_k=8,
        )
        model = PromptCapturingModel()
        await router.classify("北京天气", model=model)
        # Degraded to the old full-pool behavior instead of crashing.
        assert "weather_query" in model.last_prompt
        assert "filler_0" in model.last_prompt

    @pytest.mark.asyncio
    async def test_retriever_returning_unknown_ids_is_tolerated(self):
        class NoisyRetriever:
            async def rank(self, query, docs):
                return ["ghost_id", *docs.keys()]

        router = IntentRouter(
            scenarios=_make_scenarios(30),
            retriever=NoisyRetriever(),
            prefilter_top_k=8,
        )
        model = PromptCapturingModel()
        result = await router.classify("北京天气", model=model)
        assert result is not None
        assert "ghost_id" not in model.last_prompt


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------

class TestAgentWiring:
    def test_register_scenario_passes_triggers_to_router(self):
        from swiftagentx import Agent
        from swiftagentx.tools.scenario import ScenarioConfig, ToolChainStep

        agent = Agent()
        agent.register_scenario("weather_query", ScenarioConfig(
            name="天气查询",
            description="查询城市天气",
            triggers=["天气", "气温"],
            tool_chain=[ToolChainStep(tool="weather", query_template="$city")],
        ))
        info = agent.router._scenarios["weather_query"]
        assert info["triggers"] == ["天气", "气温"]
        assert info["slots"] == ["city"]

    def test_config_top_k_reaches_router(self):
        from swiftagentx import Agent
        from swiftagentx.models.config import SwiftAgentConfig

        agent = Agent(config=SwiftAgentConfig(scenario_prefilter_top_k=3))
        assert agent.router._prefilter_top_k == 3
