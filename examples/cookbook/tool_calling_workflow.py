"""
Tool calling workflow — custom tools, ReAct loop, error handling.

Shows how to:

  - Define a custom ``Tool`` and register it.
  - Let the ReAct loop chain tools naturally instead of hard-coding flow.
  - Surface tool failures into the loop so the model can recover.

Run::

    python examples/cookbook/tool_calling_workflow.py
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from typing import Any

from swiftagentx import (
    Agent,
    DummyModelClient,
    ModelResponse,
    SwiftAgentConfig,
    Tool,
    ToolOutput,
    ToolOutputType,
)

logging.getLogger("swiftagentx.tools.executor").setLevel(logging.ERROR)


class DemoReActModel(DummyModelClient):
    """Scripted ReAct model so this example really calls tools without an API key."""

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        prompt = messages[-1]["content"]
        if "Respond with ONLY: continuing" in prompt:
            return ModelResponse(content="continuing", model=self.model)
        if "Classify the user" in prompt:
            return ModelResponse(content="level=1", model=self.model)
        if "Use the ReAct pattern" in prompt:
            if "Error: Unknown country" in prompt:
                return ModelResponse(
                    content="Final Answer: I could not find that country in the demo database.",
                    model=self.model,
                )
            user_input = _extract_user_input(prompt).lower()
            if "capital" in user_input:
                country = "atlantis" if "atlantis" in user_input else "france"
                return ModelResponse(
                    content=(
                        "Thought: I should resolve the country to its capital.\n"
                        "Action: city_lookup\n"
                        f'Action Input: {{"country": "{country}"}}'
                    ),
                    model=self.model,
                )
            if "weather" in user_input:
                return ModelResponse(
                    content=(
                        "Thought: I should fetch the weather for the city.\n"
                        "Action: weather\n"
                        'Action Input: {"city": "Tokyo"}'
                    ),
                    model=self.model,
                )
        if "Context and tool results:" in prompt and "Error: Unknown country" in prompt:
            return ModelResponse(
                content="I could not find that country in the demo database.",
                model=self.model,
            )
        return await super().chat(messages, **kwargs)


def _extract_user_input(prompt: str) -> str:
    match = re.search(r"User input:\s*(.*?)\nIteration:", prompt, re.DOTALL)
    return match.group(1).strip() if match else prompt


class CityLookupTool(Tool):
    """Pretends to resolve a country name to a capital city."""

    def __init__(self) -> None:
        super().__init__(name="city_lookup",
                         description="Resolve a country name to its capital city.",
                         output_type=ToolOutputType.DIRECT_OUTPUT)
        self._db = {"france": "Paris", "germany": "Berlin", "japan": "Tokyo"}

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        country = (kwargs.get("query") or kwargs.get("country") or "").strip().lower()
        if country in self._db:
            return ToolOutput(
                success=True,
                result=self._db[country],
                output_type=ToolOutputType.DIRECT_OUTPUT,
            )
        return ToolOutput(success=False, result=None,
                          error=f"Unknown country: {country!r}")


class WeatherTool(Tool):
    """Pretends to fetch the current temperature for a city."""

    def __init__(self) -> None:
        super().__init__(
            name="weather",
            description="Get the temperature for a city.",
            output_type=ToolOutputType.DIRECT_OUTPUT,
        )

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        city = (kwargs.get("query") or kwargs.get("city") or "").strip()
        if not city:
            return ToolOutput(success=False, result=None, error="city is required")
        # Deterministic pseudo-random temperature so the demo is repeatable.
        rng = random.Random(hash(city) & 0xFFFFFFFF)
        temp = rng.randint(-5, 35)
        return ToolOutput(
            success=True,
            result=f"{city}: {temp}°C",
            output_type=ToolOutputType.DIRECT_OUTPUT,
        )


async def main() -> None:
    agent = Agent(
        model=DemoReActModel(api_key="demo", model="demo-react"),
        config=SwiftAgentConfig(
            name="tool-workflow",
            max_iterations=4,
            enable_cache=False,
            memory_enable_topic_change_hook=False,
        ),
    )
    agent.register_tool(CityLookupTool())
    agent.register_tool(WeatherTool())

    queries = [
        "What's the capital of France?",                 # 1 tool call
        "What's the weather like in Tokyo?",             # 1 tool call
        "Look up the capital of Atlantis.",              # tool fails — model should respond gracefully
    ]
    for q in queries:
        print(f"> {q}")
        resp = await agent.run(q, user_id="demo", session_id="tool-demo")
        print(f"< {resp.answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
