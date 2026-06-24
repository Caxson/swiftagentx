"""
SwiftAgentX Quickstart Example

This example shows how to create a basic agent with a custom tool.

Usage:
    python examples/quickstart/hello_agent.py
"""

import asyncio

from swiftagentx import Agent, DummyModelClient, Tool, ToolOutput


class EchoTool(Tool):
    """A simple echo tool for demonstration."""

    def __init__(self):
        super().__init__(
            name="echo",
            description="Echoes back the input text",
        )

    async def execute(self, context, **kwargs):
        text = kwargs.get("text", context.user_input)
        return ToolOutput(success=True, result=f"Echo: {text}")


async def main():
    # Create an agent with a dummy model (no API key needed)
    agent = Agent(
        name="HelloAgent",
        model=DummyModelClient(api_key="test", model="dummy"),
    )

    # Register a tool
    agent.register_tool(EchoTool())

    # Run the agent
    response = await agent.run("Hello, SwiftAgentX!")
    print(f"Answer: {response.answer}")
    print(f"Time: {response.execution_time_ms:.1f}ms")
    print(f"Iterations: {response.total_iterations}")


if __name__ == "__main__":
    asyncio.run(main())
