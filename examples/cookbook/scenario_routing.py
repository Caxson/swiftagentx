"""
Scenario routing — pre-defined tool chains that skip the ReAct loop.

A scenario binds an intent to a deterministic tool chain. When the LIGHT
model classifies a request as that intent, SwiftAgentX runs the chain
directly — no Thought→Action→Observation iteration, no extra LLM calls.

For high-frequency request patterns (order lookup, balance check, FAQ
match, weather, etc.) this saves 2-3 LLM calls per request.

Run::

    python examples/cookbook/scenario_routing.py
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from swiftagentx import (
    Agent,
    DummyModelClient,
    ModelResponse,
    ScenarioConfig,
    SwiftAgentConfig,
    Tool,
    ToolChainStep,
    ToolOutput,
)


class DemoRouterModel(DummyModelClient):
    """Tiny scripted classifier so the demo really exercises Scenario routing."""

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        prompt = messages[-1]["content"]
        if "Respond with ONLY: continuing" in prompt:
            return ModelResponse(content="continuing", model=self.model)
        if "Classify the user" in prompt:
            user_input = _extract_user_input(prompt).lower()
            if "weather" in user_input or "temperature" in user_input:
                return ModelResponse(
                    content='level=2 scenario=weather slots={"city": "Tokyo"}',
                    model=self.model,
                )
            if "balance" in user_input or "account" in user_input:
                return ModelResponse(
                    content='level=2 scenario=balance slots={}',
                    model=self.model,
                )
        return await super().chat(messages, **kwargs)


def _extract_user_input(prompt: str) -> str:
    match = re.search(r"User input:\s*(.*?)\n\n", prompt, re.DOTALL)
    return match.group(1) if match else prompt


class WeatherTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="weather", description="Get the weather for a city.")

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        city = kwargs.get("query") or kwargs.get("city") or "unknown"
        return ToolOutput(success=True, result=f"{city}: sunny, 24°C, gentle breeze")


class BalanceTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="balance", description="Get the user's account balance.")

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        user_id = getattr(context, "user_id", "anonymous")
        balances = {"alice": 1234.56, "bob": 42.00}
        amount = balances.get(user_id, 0.00)
        return ToolOutput(success=True, result=f"Balance for {user_id}: ${amount:.2f}")


async def main() -> None:
    agent = Agent(
        model=DemoRouterModel(api_key="demo", model="demo-router"),
        config=SwiftAgentConfig(
            name="scenario-demo",
            enable_cache=False,
            memory_enable_topic_change_hook=False,
        ),
    )
    agent.register_tool(WeatherTool())
    agent.register_tool(BalanceTool())

    agent.register_scenario(
        "weather",
        ScenarioConfig(
            name="Weather",
            description="Weather lookup",
            triggers=["weather", "temperature", "forecast"],
            tool_chain=[ToolChainStep(tool="weather", query_template="$city")],
            cache_ttl=300,
            output_type="direct",
        ),
    )
    agent.register_scenario(
        "balance",
        ScenarioConfig(
            name="Account balance",
            description="Account balance lookup",
            triggers=["balance", "how much", "account"],
            tool_chain=[ToolChainStep(tool="balance")],
            cache_ttl=30,
            output_type="direct",
        ),
    )

    interactions = [
        ("alice", "What's the weather in Tokyo?", {"city": "Tokyo"}),
        ("alice", "Check my balance please.", {}),
        ("bob",   "What's my account balance?", {}),
    ]
    for user, query, kwargs in interactions:
        print(f"> [{user}] {query}")
        resp = await agent.run(query, user_id=user, session_id=f"s-{user}", **kwargs)
        print(f"< {resp.answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
