"""
Dual-model optimization — fast LIGHT model for classification, HEAVY for reasoning.

The cost difference between a fast small model and a frontier model is
~30x. SwiftAgentX lets you use the cheap one for intent classification
(which dominates request volume) and only pay for the expensive one when
the model actually needs to *think*.

This example runs against a real OpenAI-compatible endpoint when
LLM_API_KEY is set, otherwise falls back to dummy clients.

Env vars::

    LLM_API_KEY        required for real mode
    LLM_BASE_URL       default https://api.openai.com/v1
    LLM_MODEL_LIGHT    default gpt-4o-mini
    LLM_MODEL_HEAVY    default gpt-4o

Run::

    python examples/cookbook/dual_model_optimization.py
"""

from __future__ import annotations

import asyncio
import os

from swiftagentx import Agent, DummyModelClient, ModelTier, SwiftAgentConfig


def _build_clients() -> dict[ModelTier, object]:
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        print("[demo] LLM_API_KEY not set — falling back to DummyModelClient.\n")
        return {
            ModelTier.LIGHT: DummyModelClient(api_key="demo", model="dummy-light"),
            ModelTier.HEAVY: DummyModelClient(api_key="demo", model="dummy-heavy"),
        }

    from swiftagentx.providers.openai_compatible import OpenAICompatibleProvider

    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    light_model = os.environ.get("LLM_MODEL_LIGHT", "gpt-4o-mini")
    heavy_model = os.environ.get("LLM_MODEL_HEAVY", "gpt-4o")
    return {
        ModelTier.LIGHT: OpenAICompatibleProvider(api_key=api_key, model=light_model, api_base=base),
        ModelTier.HEAVY: OpenAICompatibleProvider(api_key=api_key, model=heavy_model, api_base=base),
    }


async def main() -> None:
    agent = Agent(
        models=_build_clients(),  # type: ignore[arg-type]
        config=SwiftAgentConfig(name="dual-model", max_iterations=3),
    )

    queries = [
        "What's 2 + 2?",  # trivial — LIGHT classify, DIRECT response
        "Compare the trade-offs of REST vs gRPC for a high-throughput service.",  # forces ReAct/HEAVY
        "Hi, can you say hello?",  # trivial chitchat
    ]
    for q in queries:
        print(f"> {q}")
        resp = await agent.run(q, user_id="demo", session_id="dual-demo")
        print(f"< {resp.answer[:200]}{'…' if len(resp.answer) > 200 else ''}\n")


if __name__ == "__main__":
    asyncio.run(main())
