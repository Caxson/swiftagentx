"""
Tests for the v0.3 Hook system.

Covers:

- HookRegistry register / dispatch / unregister
- PythonHook (with and without condition)
- LLMHook with stubbed model and JSON parse
- ShellHook end-to-end via /bin/sh -c (skipped on Windows)
- Hook returning SHORT_CIRCUIT stops downstream hooks for that event
- Hook raising exception is logged + dispatch continues
- HookContext field defaults
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import pytest

from swiftagentx.core.hooks import (
    HookContext,
    HookEvent,
    HookRegistry,
    HookResult,
    LLMHook,
    PythonHook,
    ShellHook,
)


@pytest.mark.asyncio
async def test_register_and_dispatch_python_hook() -> None:
    registry = HookRegistry()
    fired: list[str] = []

    async def handler(ctx: HookContext) -> HookResult:
        fired.append(ctx.user_input)
        return HookResult()

    registry.register(
        PythonHook(
            name="record_input",
            events={HookEvent.REQUEST_START},
            handler=handler,
        )
    )
    ctx = HookContext(event=HookEvent.REQUEST_START, user_input="hi")
    result = await registry.dispatch(HookEvent.REQUEST_START, ctx)
    assert result.action == "continue"
    assert fired == ["hi"]


@pytest.mark.asyncio
async def test_condition_skips_handler() -> None:
    registry = HookRegistry()
    fired: list[str] = []

    async def handler(ctx: HookContext) -> HookResult:
        fired.append("yes")
        return HookResult()

    def condition(ctx: HookContext) -> bool:
        return ctx.user_input.startswith("/admin")

    registry.register(
        PythonHook(
            name="admin_only",
            events={HookEvent.REQUEST_START},
            handler=handler,
            condition=condition,
        )
    )

    await registry.dispatch(HookEvent.REQUEST_START,
                            HookContext(event=HookEvent.REQUEST_START, user_input="hello"))
    assert fired == []

    await registry.dispatch(HookEvent.REQUEST_START,
                            HookContext(event=HookEvent.REQUEST_START, user_input="/admin shutdown"))
    assert fired == ["yes"]


@pytest.mark.asyncio
async def test_short_circuit_stops_downstream_hooks() -> None:
    registry = HookRegistry()
    fired: list[str] = []

    async def first(ctx: HookContext) -> HookResult:
        fired.append("first")
        return HookResult(action="short_circuit", answer="from first")

    async def second(ctx: HookContext) -> HookResult:
        fired.append("second")  # should NOT run
        return HookResult()

    registry.register(PythonHook("first", {HookEvent.BEFORE_CLASSIFY}, first))
    registry.register(PythonHook("second", {HookEvent.BEFORE_CLASSIFY}, second))

    result = await registry.dispatch(
        HookEvent.BEFORE_CLASSIFY,
        HookContext(event=HookEvent.BEFORE_CLASSIFY),
    )
    assert result.action == "short_circuit"
    assert result.answer == "from first"
    assert fired == ["first"]


@pytest.mark.asyncio
async def test_exception_in_handler_does_not_break_dispatch() -> None:
    registry = HookRegistry()
    fired: list[str] = []

    async def explodes(ctx: HookContext) -> HookResult:
        raise RuntimeError("boom")

    async def survives(ctx: HookContext) -> HookResult:
        fired.append("survived")
        return HookResult()

    registry.register(PythonHook("explodes", {HookEvent.REQUEST_START}, explodes))
    registry.register(PythonHook("survives", {HookEvent.REQUEST_START}, survives))

    await registry.dispatch(HookEvent.REQUEST_START,
                            HookContext(event=HookEvent.REQUEST_START))
    assert fired == ["survived"]


@pytest.mark.asyncio
async def test_unregister_removes_hook() -> None:
    registry = HookRegistry()

    async def handler(ctx: HookContext) -> HookResult:
        return HookResult()

    registry.register(PythonHook("h", {HookEvent.REQUEST_START}, handler))
    assert "h" in registry.list_hooks()

    assert registry.unregister("h") is True
    assert "h" not in registry.list_hooks()


@pytest.mark.asyncio
async def test_llm_hook_parses_json() -> None:
    class FakeModel:
        async def chat(self, messages: list[dict[str, str]], **kw: Any) -> Any:
            class _R:
                content = json.dumps({"action": "short_circuit", "answer": "from-LLM"})
            return _R()

    class FakeAgent:
        light_model = FakeModel()

    registry = HookRegistry()
    registry.register(LLMHook(
        name="ask_llm",
        events={HookEvent.BEFORE_CLASSIFY},
        prompt_template="Decide for input: {user_input}",
    ))

    result = await registry.dispatch(
        HookEvent.BEFORE_CLASSIFY,
        HookContext(event=HookEvent.BEFORE_CLASSIFY,
                    user_input="should I short circuit?",
                    agent=FakeAgent()),
    )
    assert result.action == "short_circuit"
    assert result.answer == "from-LLM"


@pytest.mark.asyncio
async def test_llm_hook_returns_default_on_unparseable() -> None:
    class FakeModel:
        async def chat(self, messages: list[dict[str, str]], **kw: Any) -> Any:
            class _R:
                content = "this is not JSON"
            return _R()

    class FakeAgent:
        light_model = FakeModel()

    registry = HookRegistry()
    registry.register(LLMHook(
        name="ask_llm",
        events={HookEvent.REQUEST_START},
        prompt_template="anything",
    ))
    result = await registry.dispatch(
        HookEvent.REQUEST_START,
        HookContext(event=HookEvent.REQUEST_START, agent=FakeAgent()),
    )
    # Unparseable response → default HookResult with metadata.raw set.
    assert result.action == "continue"
    assert "raw" in result.metadata


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell required")
async def test_shell_hook_roundtrip() -> None:
    """ShellHook spawns a subprocess that echoes JSON on stdout."""
    registry = HookRegistry()
    # /bin/sh -c 'read in; echo {"action":"continue","metadata":{"seen":1}}'
    registry.register(ShellHook(
        name="sh_echo",
        events={HookEvent.REQUEST_START},
        command=["/bin/sh", "-c",
                 'read line; echo \'{"action":"continue","metadata":{"seen":1}}\''],
        timeout_seconds=5,
    ))
    result = await registry.dispatch(
        HookEvent.REQUEST_START,
        HookContext(event=HookEvent.REQUEST_START, user_input="hi"),
    )
    assert result.action == "continue"
    assert result.metadata.get("seen") == 1


@pytest.mark.asyncio
async def test_hook_must_declare_events() -> None:
    registry = HookRegistry()
    with pytest.raises(ValueError):
        registry.register(PythonHook(name="no-events", events=set(),
                                     handler=lambda ctx: asyncio.sleep(0)))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_list_hooks_for_event() -> None:
    registry = HookRegistry()

    async def h(ctx: HookContext) -> HookResult:
        return HookResult()

    registry.register(PythonHook("a", {HookEvent.REQUEST_START}, h))
    registry.register(PythonHook("b", {HookEvent.REQUEST_END}, h))

    assert registry.list_hooks(HookEvent.REQUEST_START) == ["a"]
    assert registry.list_hooks(HookEvent.REQUEST_END) == ["b"]
    assert sorted(registry.list_hooks()) == ["a", "b"]
