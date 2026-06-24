"""
RAG chatbot — knowledge base + multi-turn dialog with session memory.

Demonstrates how SwiftAgentX combines:

  - A pluggable knowledge base (here ``MemoryKnowledgeBase`` with TF-IDF).
  - Session-scoped memory so the model has prior turns as context.
  - The ``KnowledgeBaseStage`` pipeline stage so exact matches short-circuit
    *before* the model is even consulted.

Run::

    python examples/cookbook/rag_chatbot.py
"""

from __future__ import annotations

import asyncio

from swiftagentx import (
    Agent,
    Document,
    DummyModelClient,
    KnowledgeBaseStage,
    MemoryKnowledgeBase,
    SwiftAgentConfig,
)


async def build_agent() -> Agent:
    agent = Agent(
        model=DummyModelClient(api_key="demo", model="dummy"),
        config=SwiftAgentConfig(name="rag-bot", enable_cache=True),
    )

    kb = MemoryKnowledgeBase()
    await kb.add_documents([
        Document(doc_id="kb-1",
                 content="SwiftAgentX uses a three-level cache: KB exact match, "
                         "per-user tool result, and session variables."),
        Document(doc_id="kb-2",
                 content="The framework is licensed under Apache-2.0 and supports "
                         "Python 3.10 and newer."),
        Document(doc_id="kb-3",
                 content="Scenarios skip the ReAct loop, saving 2-3 LLM calls per "
                         "request when the intent maps to a known pattern."),
    ])
    agent.set_knowledge_base(kb)

    # Mount the KB as a pipeline stage so local-dev FAQ matches return before
    # intent classification. Tune this threshold against your real KB data in
    # production; the tiny TF-IDF demo corpus needs a lower value.
    agent.pipeline.add_stage(KnowledgeBaseStage(kb=kb, threshold=0.35))

    return agent


async def main() -> None:
    agent = await build_agent()
    user = "demo-user"
    session = "rag-session"

    questions = [
        "How many cache levels does SwiftAgentX use?",
        "What license is it under?",
        "Do scenarios use the ReAct loop?",
        "Summarize what you just told me.",
    ]

    for q in questions:
        print(f"> {q}")
        resp = await agent.run(q, user_id=user, session_id=session)
        print(f"< {resp.answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
