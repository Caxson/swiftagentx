"""
Customer service agent — KB short-circuit + scenario fallback + scripted tools.

Demonstrates the three execution paths SwiftAgentX is designed for, in
descending order of latency:

  1. Knowledge-base exact match  -> 0 LLM calls, ~0 ms
  2. Scenario tool chain          -> 1 LIGHT-model call, no ReAct loop
  3. Direct LLM response          -> 1 HEAVY-model call

Run::

    python examples/cookbook/customer_service_agent.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from swiftagentx import (
    Agent,
    Document,
    DummyModelClient,
    MemoryKnowledgeBase,
    ScenarioConfig,
    SwiftAgentConfig,
    Tool,
    ToolChainStep,
    ToolOutput,
)


class OrderStatusTool(Tool):
    """A pretend order-status lookup. Replace with a real API call in production."""

    def __init__(self) -> None:
        super().__init__(name="order_status", description="Look up an order's status by ID.")
        self._orders = {
            "A100": "shipped, ETA 2026-05-28",
            "A101": "in warehouse, awaiting carrier pickup",
        }

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        order_id = (kwargs.get("query") or kwargs.get("order_id") or "").upper().strip()
        if order_id in self._orders:
            return ToolOutput(success=True, result=f"Order {order_id}: {self._orders[order_id]}")
        return ToolOutput(success=False, result=None, error=f"Order {order_id!r} not found")


async def build_agent() -> Agent:
    agent = Agent(
        model=DummyModelClient(api_key="demo", model="dummy"),
        config=SwiftAgentConfig(name="customer-service", max_iterations=3, enable_cache=True),
    )

    # 1. Knowledge base: exact matches return instantly with zero LLM calls.
    kb = MemoryKnowledgeBase()
    await kb.add_documents([
        Document(doc_id="faq-returns",
                 content="Returns are accepted within 7 days of delivery."),
        Document(doc_id="faq-shipping",
                 content="Standard shipping takes 3-5 business days."),
        Document(doc_id="faq-refund",
                 content="Refunds are processed within 5 business days."),
    ])
    agent.set_knowledge_base(kb)

    # 2. Scenario toolchain: high-frequency request pattern -> tool, no ReAct.
    agent.register_tool(OrderStatusTool())
    agent.register_scenario(
        "order_status",
        ScenarioConfig(
            name="Order Status",
            description="Lookup order shipping status by ID.",
            triggers=["order", "where is my", "status", "shipment"],
            tool_chain=[ToolChainStep(tool="order_status", query_template="$order_id")],
            cache_ttl=120,
            output_type="direct",
        ),
    )

    return agent


async def main() -> None:
    agent = await build_agent()
    user = "demo-user"

    # KB short-circuit path
    print("> Returns are accepted within 7 days of delivery.")
    resp = await agent.run(
        "Returns are accepted within 7 days of delivery.", user_id=user, session_id="s1"
    )
    print(f"< {resp.answer}\n")

    # Scenario path
    print("> Where is my order A100?")
    resp = await agent.run(
        "Where is my order A100?", user_id=user, session_id="s1", order_id="A100"
    )
    print(f"< {resp.answer}\n")

    # Direct LLM fallback
    print("> Tell me a joke about logistics.")
    resp = await agent.run("Tell me a joke about logistics.", user_id=user, session_id="s1")
    print(f"< {resp.answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
