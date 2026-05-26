"""
Tests for core modules: memory, prompt, parameter, router, pipeline.
"""

import pytest

from swiftagentx import DummyModelClient
from swiftagentx.core.memory import SessionMemory
from swiftagentx.core.parameter import ParameterManager
from swiftagentx.core.pipeline import PipelineStage, RequestPipeline, StageAction, StageResult
from swiftagentx.core.prompt import PromptManager
from swiftagentx.core.router import IntentLevel, IntentRouter


class TestSessionMemory:
    def test_add_and_get_messages(self):
        mem = SessionMemory()
        mem.add_message("user", "hello")
        mem.add_message("assistant", "hi")
        assert len(mem) == 2

    def test_conversation_text(self):
        mem = SessionMemory()
        mem.add_message("user", "question")
        mem.add_message("assistant", "answer")
        text = mem.get_conversation_text()
        assert "USER: question" in text
        assert "ASSISTANT: answer" in text

    def test_max_messages_cleanup(self):
        mem = SessionMemory(max_messages=3)
        for i in range(5):
            mem.add_message("user", f"msg {i}")
        assert len(mem) == 3

    def test_export(self):
        mem = SessionMemory()
        mem.add_message("user", "test")
        exported = mem.export()
        assert len(exported["messages"]) == 1
        assert exported["messages"][0]["role"] == "user"


class TestPromptManager:
    def test_default_prompts(self):
        pm = PromptManager()
        assert pm.get("system_role") is not None
        assert pm.get("react_thought") is not None

    def test_register_and_render(self):
        pm = PromptManager()
        pm.register("test", "Hello, {name}!")
        result = pm.render("test", name="World")
        assert result == "Hello, World!"

    def test_render_missing_template_raises(self):
        pm = PromptManager()
        with pytest.raises(ValueError, match="not found"):
            pm.render("nonexistent")

    def test_render_safe_fallback(self):
        pm = PromptManager()
        result = pm.render_safe("nonexistent", default="fallback")
        assert result == "fallback"

    def test_get_output_prompt(self):
        pm = PromptManager()
        prompt = pm.get_output_prompt()
        assert len(prompt) > 0


class TestParameterManager:
    def test_global_params(self):
        pm = ParameterManager()
        pm.set_global_param("key", "value")
        assert pm.get_global_param("key") == "value"

    def test_session_params(self):
        pm = ParameterManager()
        pm.set_session_param("s1", "key", "val")
        assert pm.get_session_param("s1", "key") == "val"

    def test_merged_params(self):
        pm = ParameterManager()
        pm.set_global_param("g_key", "global")
        pm.set_session_param("s1", "s_key", "session")
        merged = pm.get_merged_params("s1")
        assert merged["g_key"] == "global"
        assert merged["s_key"] == "session"

    def test_session_override_global(self):
        pm = ParameterManager()
        pm.set_global_param("key", "global")
        pm.set_session_param("s1", "key", "session")
        merged = pm.get_merged_params("s1")
        assert merged["key"] == "session"


class TestIntentRouter:
    def test_parse_level_1(self):
        router = IntentRouter()
        result = router._parse_classification("level=1")
        assert result.level == IntentLevel.REACT

    def test_parse_level_2_with_scenario(self):
        router = IntentRouter(scenarios={"weather": {"name": "Weather"}})
        result = router._parse_classification("level=2 scenario=weather")
        assert result.level == IntentLevel.SCENARIO
        assert result.scenario == "weather"

    def test_parse_level_3(self):
        router = IntentRouter()
        result = router._parse_classification("level=3")
        assert result.level == IntentLevel.DIRECT

    @pytest.mark.asyncio
    async def test_classify_without_model(self):
        router = IntentRouter()
        result = await router.classify("hello")
        assert result.level == IntentLevel.DIRECT

    @pytest.mark.asyncio
    async def test_classify_with_model(self):
        router = IntentRouter()
        model = DummyModelClient(api_key="test", model="dummy")
        result = await router.classify("hello", model=model)
        assert result.level is not None


class TestPipeline:
    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        pipeline = RequestPipeline()
        result = await pipeline.execute({})
        assert result.action == StageAction.CONTINUE

    @pytest.mark.asyncio
    async def test_pipeline_with_stage(self):
        class TestStage(PipelineStage):
            async def execute(self, context):
                context["processed"] = True
                return StageResult(data={"processed": True})

        pipeline = RequestPipeline()
        pipeline.add_stage(TestStage("test"))
        ctx = {}
        result = await pipeline.execute(ctx)
        assert ctx["processed"] is True

    @pytest.mark.asyncio
    async def test_pipeline_short_circuit(self):
        class CacheStage(PipelineStage):
            async def execute(self, context):
                return StageResult(
                    action=StageAction.SHORT_CIRCUIT,
                    answer="cached",
                )

        class NeverReached(PipelineStage):
            async def execute(self, context):
                raise AssertionError("Should not be reached")

        pipeline = RequestPipeline()
        pipeline.add_stage(CacheStage("cache"))
        pipeline.add_stage(NeverReached("never"))
        result = await pipeline.execute({})
        assert result.action == StageAction.SHORT_CIRCUIT
        assert result.answer == "cached"
