"""Search and fetch through registered MCP tools, without an API key or LLM.

Run: pip install -e '.[openai]'
     python examples/cookbook/parallel_search.py
Requires internet access; anonymous Parallel usage is rate limited.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from swiftagentx import Agent, DummyModelClient


async def main() -> None:
    # No model inference is performed; invoke the registered tools directly.
    agent = Agent(model=DummyModelClient(api_key="unused", model="dummy"))
    session_id = str(uuid4())
    context = SimpleNamespace(variables={}, user_input="MCP documentation", session_id=session_id)
    try:
        names = await agent.register_mcp_server(
            "parallel",
            transport="streamable_http",
            url="https://search.parallel.ai/mcp",
        )
        print("Registered:", ", ".join(names))
        calls = [
            (
                "parallel.web_search",
                {
                    "objective": "Find the official Model Context Protocol documentation",
                    "search_queries": ["Model Context Protocol official documentation"],
                    "session_id": session_id,
                },
            ),
            (
                "parallel.web_fetch",
                {
                    "urls": ["https://modelcontextprotocol.io/introduction"],
                    "objective": "What is the Model Context Protocol?",
                    "session_id": session_id,
                },
            ),
        ]
        for name, arguments in calls:
            tool = agent.tool_registry.get(name)
            if tool is None:
                raise RuntimeError(f"MCP server did not expose {name}")
            output = await tool.execute(context, **arguments)
            if not output.success:
                raise RuntimeError(output.error)
            print(f"\n{name}:\n{output.result}")
    finally:
        await agent.shutdown_mcp_servers()


if __name__ == "__main__":
    asyncio.run(main())
