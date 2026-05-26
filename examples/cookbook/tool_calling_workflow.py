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
import random
from typing import Any

from swiftagentx import (
    Agent,
    DummyModelClient,
    SwiftAgentConfig,
    Tool,
    ToolOutput,
)


class CityLookupTool(Tool):
    """Pretends to resolve a country name to a capital city."""

    def __init__(self) -> None:
        super().__init__(name="city_lookup",
                         description="Resolve a country name to its capital city.")
        self._db = {"france": "Paris", "germany": "Berlin", "japan": "Tokyo"}

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        country = (kwargs.get("query") or kwargs.get("country") or "").strip().lower()
        if country in self._db:
            return ToolOutput(success=True, result=self._db[country])
        return ToolOutput(success=False, result=None,
                          error=f"Unknown country: {country!r}")


class WeatherTool(Tool):
    """Pretends to fetch the current temperature for a city."""

    def __init__(self) -> None:
        super().__init__(name="weather", description="Get the temperature for a city.")

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        city = (kwargs.get("query") or kwargs.get("city") or "").strip()
        if not city:
            return ToolOutput(success=False, result=None, error="city is required")
        # Deterministic pseudo-random temperature so the demo is repeatable.
        rng = random.Random(hash(city) & 0xFFFFFFFF)
        temp = rng.randint(-5, 35)
        return ToolOutput(success=True, result=f"{city}: {temp}°C")


async def main() -> None:
    agent = Agent(
        model=DummyModelClient(api_key="demo", model="dummy"),
        config=SwiftAgentConfig(name="tool-workflow", max_iterations=4),
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
