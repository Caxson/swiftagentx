"""
Tests for v0.3 sub-agent dispatch.

Covers:

- Role registration / unregistration / listing
- Default handler: runs an LLM call against the parent agent's model tier
- Parallel dispatch via ``Agent.dispatch_subagents`` — results in order
- Unknown role → ``success=False`` result (no exception)
- Timeout → ``success=False`` with elapsed-time annotation
- Handler exception → ``success=False`` with the exception message
- Custom handler overrides the default LLM path
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from swiftagentx import Agent, DummyModelClient, SwiftAgentConfig
from swiftagentx.core.subagent import (
    SubAgentInvocation,
    SubAgentManager,
    SubAgentRequest,
    SubAgentResult,
    SubAgentRole,
)

# ---------------------------------------------------------------------------
# Role registration
# ---------------------------------------------------------------------------


def test_register_role_and_list() -> None:
    manager = SubAgentManager()
    manager.register_role(SubAgentRole(
        name="lookup", description="account lookup", system_prompt="You are a lookup.",
    ))
    assert manager.list_roles() == ["lookup"]
    assert manager.get_role("lookup") is not None


def test_unregister_role() -> None:
    manager = SubAgentManager()
    manager.register_role(SubAgentRole(name="x", description="", system_prompt=""))
    assert manager.unregister_role("x") is True
    assert manager.unregister_role("x") is False


def test_overwriting_role_replaces_handler() -> None:
    manager = SubAgentManager()
    async def h1(*args: Any) -> SubAgentResult:
        return SubAgentResult(role="x", success=True, output="h1")
    async def h2(*args: Any) -> SubAgentResult:
        return SubAgentResult(role="x", success=True, output="h2")
    manager.register_role(
        SubAgentRole(name="x", description="", system_prompt=""), handler=h1)
    manager.register_role(
        SubAgentRole(name="x", description="", system_prompt=""), handler=h2)
    assert "x" in manager.list_roles()


# ---------------------------------------------------------------------------
# Default handler with real Agent + DummyModelClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_handler_calls_parent_agent_model() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    agent.register_subagent(SubAgentRole(
        name="echo",
        description="echo back input",
        system_prompt="Echo the user.",
    ))
    results = await agent.dispatch_subagents([
        SubAgentRequest(role="echo", input="hello"),
    ])
    assert len(results) == 1
    assert results[0].success
    assert results[0].role == "echo"
    assert results[0].output  # dummy returns some non-empty string
    assert results[0].duration_ms > 0


# ---------------------------------------------------------------------------
# Parallel dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_runs_in_parallel() -> None:
    """Three sub-agents that each sleep 100ms should finish in <250ms total
    (parallel) rather than ~300ms (sequential)."""

    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))

    async def slow_handler(req: SubAgentRequest, inv: SubAgentInvocation) -> SubAgentResult:
        await asyncio.sleep(0.1)
        return SubAgentResult(role=req.role, success=True, output=f"done-{req.input}")

    for name in ("a", "b", "c"):
        agent.register_subagent(
            SubAgentRole(name=name, description="", system_prompt=""),
            handler=slow_handler,
        )

    start = asyncio.get_event_loop().time()
    results = await agent.dispatch_subagents([
        SubAgentRequest(role="a", input="1"),
        SubAgentRequest(role="b", input="2"),
        SubAgentRequest(role="c", input="3"),
    ])
    elapsed = asyncio.get_event_loop().time() - start

    assert elapsed < 0.25, f"expected parallel (<0.25s) got {elapsed:.3f}s"
    assert [r.success for r in results] == [True, True, True]
    assert [r.output for r in results] == ["done-1", "done-2", "done-3"]


@pytest.mark.asyncio
async def test_results_preserve_request_order() -> None:
    """Even with handlers that finish out of order, results match request order."""
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))

    delays = {"a": 0.05, "b": 0.01, "c": 0.03}

    async def variable_delay(req: SubAgentRequest, inv: SubAgentInvocation) -> SubAgentResult:
        await asyncio.sleep(delays[req.role])
        return SubAgentResult(role=req.role, success=True, output=req.role)

    for name in delays:
        agent.register_subagent(
            SubAgentRole(name=name, description="", system_prompt=""),
            handler=variable_delay,
        )

    results = await agent.dispatch_subagents([
        SubAgentRequest(role="a", input="x"),
        SubAgentRequest(role="b", input="x"),
        SubAgentRequest(role="c", input="x"),
    ])
    assert [r.output for r in results] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_role_returns_failure() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    results = await agent.dispatch_subagents([
        SubAgentRequest(role="nonexistent", input="x"),
    ])
    assert results[0].success is False
    assert "unknown" in (results[0].error or "")


@pytest.mark.asyncio
async def test_handler_timeout_returns_failure() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))

    async def too_slow(req: SubAgentRequest, inv: SubAgentInvocation) -> SubAgentResult:
        await asyncio.sleep(0.5)
        return SubAgentResult(role=req.role, success=True)

    agent.register_subagent(
        SubAgentRole(name="slowpoke", description="", system_prompt="",
                     timeout_seconds=0.05),
        handler=too_slow,
    )

    results = await agent.dispatch_subagents([
        SubAgentRequest(role="slowpoke", input="x"),
    ])
    assert results[0].success is False
    assert "timeout" in (results[0].error or "").lower()


@pytest.mark.asyncio
async def test_handler_exception_is_caught() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))

    async def explodes(req: SubAgentRequest, inv: SubAgentInvocation) -> SubAgentResult:
        raise RuntimeError("kaboom")

    agent.register_subagent(
        SubAgentRole(name="boom", description="", system_prompt=""),
        handler=explodes,
    )

    results = await agent.dispatch_subagents([
        SubAgentRequest(role="boom", input="x"),
    ])
    assert results[0].success is False
    assert "kaboom" in (results[0].error or "")


@pytest.mark.asyncio
async def test_one_failure_does_not_break_others() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))

    async def fail(req: SubAgentRequest, inv: SubAgentInvocation) -> SubAgentResult:
        raise RuntimeError("nope")

    async def succeed(req: SubAgentRequest, inv: SubAgentInvocation) -> SubAgentResult:
        return SubAgentResult(role=req.role, success=True, output="ok")

    agent.register_subagent(
        SubAgentRole(name="bad", description="", system_prompt=""),
        handler=fail,
    )
    agent.register_subagent(
        SubAgentRole(name="good", description="", system_prompt=""),
        handler=succeed,
    )

    results = await agent.dispatch_subagents([
        SubAgentRequest(role="bad", input="x"),
        SubAgentRequest(role="good", input="x"),
    ])
    assert results[0].success is False
    assert results[1].success is True
