"""
Tests for pipeline execution inside Agent.run().

Regression coverage for the v0.2.0 fix where ``Agent.pipeline`` was created
but never actually executed during request handling. After the fix:

- ``Agent.run()`` invokes ``self.pipeline.execute()`` before the cache check
- A stage that returns SHORT_CIRCUIT bypasses classification and the LLM
- ``Agent.set_knowledge_base()`` auto-installs a ``KnowledgeBaseStage``
"""

from __future__ import annotations

from typing import Any

import pytest

from swiftagentx import (
    Agent,
    Document,
    DummyModelClient,
    KnowledgeBaseStage,
    MemoryKnowledgeBase,
    PipelineStage,
    StageResult,
    SwiftAgentConfig,
)
from swiftagentx.core.pipeline import StageAction


class AlwaysShortCircuitStage(PipelineStage):
    def __init__(self, answer: str = "stage answer") -> None:
        super().__init__(name="AlwaysShortCircuit")
        self.answer = answer
        self.calls = 0

    async def execute(self, context: dict[str, Any]) -> StageResult:
        self.calls += 1
        return StageResult(action=StageAction.SHORT_CIRCUIT, answer=self.answer)


class CountingStage(PipelineStage):
    def __init__(self) -> None:
        super().__init__(name="Counting")
        self.calls = 0

    async def execute(self, context: dict[str, Any]) -> StageResult:
        self.calls += 1
        return StageResult(action=StageAction.CONTINUE)


@pytest.mark.asyncio
async def test_short_circuit_stage_skips_llm() -> None:
    agent = Agent(
        model=DummyModelClient(api_key="test", model="dummy"),
        config=SwiftAgentConfig(enable_cache=False),
    )
    stage = AlwaysShortCircuitStage(answer="hello from pipeline")
    agent.pipeline.add_stage(stage)

    response = await agent.run("any input", user_id="u1", session_id="s1")

    assert response.answer == "hello from pipeline"
    assert response.metadata.get("pipeline_short_circuit") is True
    assert stage.calls == 1


@pytest.mark.asyncio
async def test_continue_stage_does_not_short_circuit() -> None:
    agent = Agent(
        model=DummyModelClient(api_key="test", model="dummy"),
        config=SwiftAgentConfig(enable_cache=False),
    )
    stage = CountingStage()
    agent.pipeline.add_stage(stage)

    response = await agent.run("hello", user_id="u1", session_id="s1")

    assert stage.calls == 1
    assert response.metadata.get("pipeline_short_circuit") is not True


@pytest.mark.asyncio
async def test_set_knowledge_base_auto_installs_stage() -> None:
    agent = Agent(
        model=DummyModelClient(api_key="test", model="dummy"),
        config=SwiftAgentConfig(enable_cache=False),
    )
    kb = MemoryKnowledgeBase()
    await kb.add_documents([
        Document(doc_id="faq-1", content="Returns are accepted within 7 days."),
    ])

    assert agent.pipeline.list_stages() == []
    agent.set_knowledge_base(kb)

    stages = agent.pipeline.list_stages()
    assert "KnowledgeBaseStage" in stages

    response = await agent.run(
        "Returns are accepted within 7 days.", user_id="u1", session_id="s1"
    )
    assert response.metadata.get("pipeline_short_circuit") is True
    assert "Returns are accepted within 7 days" in response.answer


@pytest.mark.asyncio
async def test_set_knowledge_base_idempotent() -> None:
    """Calling set_knowledge_base twice should not duplicate the stage."""
    agent = Agent(
        model=DummyModelClient(api_key="test", model="dummy"),
        config=SwiftAgentConfig(),
    )
    kb = MemoryKnowledgeBase()
    await kb.add_documents([Document(doc_id="d", content="x")])

    agent.set_knowledge_base(kb)
    agent.set_knowledge_base(kb)

    stages = agent.pipeline.list_stages()
    assert stages.count("KnowledgeBaseStage") == 1


@pytest.mark.asyncio
async def test_set_knowledge_base_opt_out() -> None:
    """auto_short_circuit=False should skip the auto-install."""
    agent = Agent(
        model=DummyModelClient(api_key="test", model="dummy"),
        config=SwiftAgentConfig(),
    )
    kb = MemoryKnowledgeBase()
    agent.set_knowledge_base(kb, auto_short_circuit=False)
    assert agent.pipeline.list_stages() == []


@pytest.mark.asyncio
async def test_manual_kb_stage_still_works() -> None:
    """Users can still add KnowledgeBaseStage manually with a custom threshold."""
    agent = Agent(
        model=DummyModelClient(api_key="test", model="dummy"),
        config=SwiftAgentConfig(),
    )
    kb = MemoryKnowledgeBase()
    await kb.add_documents([Document(doc_id="d", content="hello")])
    agent.pipeline.add_stage(KnowledgeBaseStage(kb=kb, threshold=0.5))

    response = await agent.run("hello", user_id="u1", session_id="s1")
    assert response.metadata.get("pipeline_short_circuit") is True
