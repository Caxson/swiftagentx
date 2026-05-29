"""
Tests for the v0.3 trio: Workspace, PromptLayout, and lazy tool loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from swiftagentx import Agent, DummyModelClient, SwiftAgentConfig, Tool, ToolOutput
from swiftagentx.core.prompt_layout import PromptLayout
from swiftagentx.core.workspace import (
    InMemoryWorkspaceBackend,
    LocalDiskWorkspaceBackend,
    use_workspace,
)

# ===========================================================================
# Workspace
# ===========================================================================


@pytest.mark.asyncio
async def test_local_disk_workspace_write_and_read(tmp_path: Path) -> None:
    backend = LocalDiskWorkspaceBackend(root=tmp_path)
    async with use_workspace(backend, "s1") as ws:
        await ws.write("report.txt", "hello")
        assert await ws.read("report.txt") == b"hello"
        assert await ws.exists("report.txt")
        listing = await ws.list()
        assert "report.txt" in listing


@pytest.mark.asyncio
async def test_local_disk_workspace_path_escape_blocked(tmp_path: Path) -> None:
    backend = LocalDiskWorkspaceBackend(root=tmp_path)
    async with use_workspace(backend, "s1") as ws:
        with pytest.raises(ValueError):
            ws.path("../escape.txt")


@pytest.mark.asyncio
async def test_local_disk_workspace_session_isolation(tmp_path: Path) -> None:
    backend = LocalDiskWorkspaceBackend(root=tmp_path)
    async with use_workspace(backend, "alpha") as a:
        await a.write("file.txt", "alpha-data")
    async with use_workspace(backend, "beta") as b:
        assert await b.read("file.txt") is None
        await b.write("file.txt", "beta-data")
    # The two sessions live in separate subdirs.
    async with use_workspace(backend, "alpha") as a2:
        assert (await a2.read("file.txt")) == b"alpha-data"


@pytest.mark.asyncio
async def test_local_disk_workspace_cleanup(tmp_path: Path) -> None:
    backend = LocalDiskWorkspaceBackend(root=tmp_path)
    async with use_workspace(backend, "ephemeral", cleanup_on_exit=True) as ws:
        await ws.write("scratch.bin", b"x")
    # After cleanup the dir should be gone (or rebuilt empty on reopen).
    async with use_workspace(backend, "ephemeral") as ws2:
        assert await ws2.list() == []


@pytest.mark.asyncio
async def test_in_memory_workspace_round_trip() -> None:
    backend = InMemoryWorkspaceBackend()
    async with use_workspace(backend, "s") as ws:
        await ws.write("a", b"\x00\x01")
        await ws.write("b", "text")
        assert await ws.exists("a")
        assert await ws.exists("b")
        assert (await ws.list()) == ["a", "b"]
        assert await ws.read("a") == b"\x00\x01"
        assert await ws.read("b") == b"text"
        assert await ws.remove("a") is True
        assert await ws.remove("a") is False


@pytest.mark.asyncio
async def test_agent_workspace_context_manager(tmp_path: Path) -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    agent.workspace_backend = LocalDiskWorkspaceBackend(root=tmp_path)
    async with agent.workspace("sess") as ws:
        await ws.write("out.txt", "ok")
        assert await ws.read("out.txt") == b"ok"


# ===========================================================================
# PromptLayout
# ===========================================================================


def test_prompt_layout_chat_order() -> None:
    layout = PromptLayout(
        tools_section="TOOLS",
        system_section="SYSTEM",
        l4_summary="SUMMARY",
        l3_reference="REF",
        l2_recent_dialog=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        l1_current_input="now what?",
    )
    msgs = layout.as_chat_messages()
    # First message is the stable prefix system message.
    assert msgs[0]["role"] == "system"
    assert "TOOLS" in msgs[0]["content"]
    assert "SYSTEM" in msgs[0]["content"]
    assert "SUMMARY" in msgs[0]["content"]
    assert "REF" in msgs[0]["content"]
    # Then L2 dialog turns.
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2] == {"role": "assistant", "content": "hello"}
    # Finally the new user input.
    assert msgs[-1] == {"role": "user", "content": "now what?"}


def test_prompt_layout_skips_empty_sections() -> None:
    layout = PromptLayout(l1_current_input="hello?")
    msgs = layout.as_chat_messages()
    # No prefix system message when all stable sections are empty.
    assert len(msgs) == 1
    assert msgs[0] == {"role": "user", "content": "hello?"}


def test_prompt_layout_single_string() -> None:
    layout = PromptLayout(
        system_section="SYS",
        l1_current_input="ask",
    )
    s = layout.as_single_prompt()
    assert "SYS" in s
    assert "ask" in s
    assert s.index("SYS") < s.index("ask")


@pytest.mark.asyncio
async def test_prompt_layout_from_agent(tmp_path: Path) -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    mem = await agent.memory.get("s", "u")
    await mem.add_turn("first user msg", "first reply")
    await mem.add_turn("second user msg", "second reply")

    layout = PromptLayout.from_agent(
        agent=agent, memory=mem, user_input="now what?",
        system_section="You are helpful.",
    )
    msgs = layout.as_chat_messages()
    # System prefix should include the system_section. L2 should have 4
    # entries (2 turns × user+assistant). L1 is the final user.
    assert msgs[0]["role"] == "system"
    assert "helpful" in msgs[0]["content"]
    assert msgs[-1]["content"] == "now what?"


# ===========================================================================
# Lazy tool loading
# ===========================================================================


class _T(Tool):
    def __init__(self, name: str, description: str = "",
                 category: str = "general"):
        super().__init__(name=name, description=description, category=category)

    async def execute(self, context: Any, **kwargs: Any) -> ToolOutput:
        return ToolOutput(success=True, result=self.name)


def test_select_below_threshold_returns_all() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    for i in range(5):
        agent.register_tool(_T(name=f"tool_{i}"))
    selected = agent.tool_registry.select_tools_for_query("anything", threshold=20)
    assert len(selected) == 5


def test_select_above_threshold_filters_by_relevance() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    for name, description, category in [
        ("weather", "Get current weather for a city", "weather"),
        ("forecast", "Multi-day weather forecast", "weather"),
        ("ping", "Health-check a server", "ops"),
        ("logs", "Tail server logs", "ops"),
        ("send_email", "Send a transactional email", "comms"),
        ("read_email", "Read an inbox", "comms"),
        ("translate", "Translate a sentence", "lang"),
        ("summarize", "Summarize a long document", "lang"),
        ("currency", "Get currency exchange rate", "finance"),
        ("stock", "Get stock quote", "finance"),
        ("calendar", "List calendar events", "schedule"),
        ("flight", "Search flights", "travel"),
        ("hotel", "Search hotels", "travel"),
        ("recipe", "Find a recipe", "food"),
        ("review", "Read product reviews", "shop"),
        ("compare", "Compare products", "shop"),
        ("checkout", "Process checkout", "shop"),
        ("refund", "Process refund", "shop"),
        ("track", "Track package", "shop"),
        ("warranty", "Lookup warranty status", "shop"),
        ("install", "Show installation steps", "support"),
    ]:
        agent.register_tool(_T(name=name, description=description, category=category))

    assert agent.tool_registry.count() >= 20
    selected = agent.tool_registry.select_tools_for_query(
        "what is the weather in Beijing", threshold=15, max_returned=5,
    )
    names = [t.name for t in selected]
    assert "weather" in names
    assert "forecast" in names
    # Unrelated tools should not appear in the top-N when relevant ones exist.
    assert "currency" not in names


def test_schemas_for_query_returns_filtered_schemas() -> None:
    agent = Agent(model=DummyModelClient(api_key="k", model="d"),
                  config=SwiftAgentConfig(memory_enable_topic_change_hook=False))
    # Give each tool a unique discriminating token in its name+description.
    descriptors = [
        ("weather_now", "weather lookup"),
        ("forecast", "weather forecast for tomorrow"),
        ("currency", "currency exchange rate"),
        ("stock", "stock quote"),
        ("calendar", "calendar events listing"),
        ("flight", "flight search"),
        ("hotel", "hotel search"),
        ("recipe", "cooking recipe lookup"),
        ("review", "product review fetcher"),
        ("compare", "comparison engine"),
        ("checkout", "checkout processor"),
        ("refund", "refund processing"),
        ("track", "tracking package"),
        ("warranty", "warranty status"),
        ("translate", "translation engine"),
        ("summarize", "summarization tool"),
        ("send_email", "send email"),
        ("read_email", "read inbox"),
        ("logs", "tail server logs"),
        ("ping", "ping server health"),
        ("install", "installation steps"),
        ("uninstall", "uninstall utility"),
    ]
    for name, description in descriptors:
        agent.register_tool(_T(name=name, description=description))

    schemas = agent.tool_registry.schemas_for_query(
        "weather forecast for tomorrow", threshold=10, max_returned=3,
    )
    assert "forecast" in schemas
    assert "weather_now" in schemas
    assert len(schemas) <= 3
