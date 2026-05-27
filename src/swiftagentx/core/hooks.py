"""
Hook system for SwiftAgentX v0.3.

Three orthogonal concepts:

1. **Events** — *when* a hook fires.

   Lifecycle events fire at fixed points in ``Agent.run()``
   (``RequestStart``, ``BeforeClassify``, ``BeforeToolCall``, …).
   Semantic events fire when a condition the framework can't statically
   place evaluates true (``TopicChange``, ``ToolFailureCascade``).

2. **Conditions** — *which* hooks of a given event fire on a given turn.

   Every hook can opt out of firing by returning False from
   ``should_fire(context)``. The default is to always fire.

3. **Handlers** — *what happens* when a hook fires.

   The framework ships four handler kinds:

   - ``PythonHook``  — coroutine ``handler(context)``
   - ``ShellHook``   — shell command, JSON stdin/stdout (Claude-Code style)
   - ``LLMHook``     — one-shot LLM call whose output drives the next step
   - ``SkillHook``   — invokes a markdown-defined Skill (registered later)

A hook can return a ``HookResult`` whose ``action`` field lets it
short-circuit the request, abort it, or rewrite the context for
downstream stages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class HookEvent(str, Enum):
    """Canonical event names hooks can subscribe to.

    Lifecycle events fire at a deterministic point in ``Agent.run()``.
    Semantic events are evaluated by ``HookRegistry.fire_semantic()``
    which calls each registered hook's ``should_fire`` and runs the ones
    that return True.
    """

    # --- Lifecycle events --------------------------------------------------

    SESSION_START = "session_start"           # first time we see a session id
    REQUEST_START = "request_start"           # every request
    BEFORE_CLASSIFY = "before_classify"
    AFTER_CLASSIFY = "after_classify"
    BEFORE_SCENARIO_STEP = "before_scenario_step"
    AFTER_SCENARIO_STEP = "after_scenario_step"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_REACT_ITER = "before_react_iter"
    AFTER_REACT_ITER = "after_react_iter"
    BEFORE_RESPOND = "before_respond"
    REQUEST_END = "request_end"

    # --- Semantic events ---------------------------------------------------

    TOPIC_CHANGE = "topic_change"
    TOOL_FAILURE_CASCADE = "tool_failure_cascade"
    MAX_ITERATIONS_APPROACHING = "max_iterations_approaching"
    CACHE_HIT_RATE_LOW = "cache_hit_rate_low"

    @classmethod
    def lifecycle_events(cls) -> set[HookEvent]:
        return {
            cls.SESSION_START, cls.REQUEST_START,
            cls.BEFORE_CLASSIFY, cls.AFTER_CLASSIFY,
            cls.BEFORE_SCENARIO_STEP, cls.AFTER_SCENARIO_STEP,
            cls.BEFORE_TOOL_CALL, cls.AFTER_TOOL_CALL,
            cls.BEFORE_REACT_ITER, cls.AFTER_REACT_ITER,
            cls.BEFORE_RESPOND, cls.REQUEST_END,
        }

    @classmethod
    def semantic_events(cls) -> set[HookEvent]:
        return {
            cls.TOPIC_CHANGE, cls.TOOL_FAILURE_CASCADE,
            cls.MAX_ITERATIONS_APPROACHING, cls.CACHE_HIT_RATE_LOW,
        }


# ---------------------------------------------------------------------------
# Context + result
# ---------------------------------------------------------------------------


@dataclass
class HookContext:
    """Mutable bag of state passed to hook handlers.

    The framework populates the standard fields; handlers may set
    arbitrary keys in ``extra`` and read them later.
    """

    event: HookEvent
    user_input: str = ""
    user_id: str = ""
    session_id: str = ""
    agent: Any = None  # the Agent instance, opaque to avoid an import cycle
    memory: Any = None  # the LayeredMemory for this session, or None
    intent: Any = None  # IntentResult, populated after classify
    scenario: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    react_iteration: int | None = None
    answer: str | None = None
    error: BaseException | None = None
    extra: dict[str, Any] = field(default_factory=dict)


HookAction = Literal["continue", "short_circuit", "abort", "rewrite"]


class HookResult(BaseModel):
    """What a handler returns; tells the framework what to do next."""

    action: HookAction = "continue"
    answer: str | None = None              # used by short_circuit
    rewrite: dict[str, Any] = {}           # used by rewrite (e.g. mutate tool_args)
    metadata: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Hook base classes
# ---------------------------------------------------------------------------


class Hook:
    """Common base for all hook types.

    A hook subscribes to one or more events. ``HookRegistry.dispatch``
    calls ``should_fire`` (cheap) before ``handle`` (expensive). Override
    only the methods you need.
    """

    name: str = ""
    events: set[HookEvent] = set()

    def __init__(self, name: str = "", events: set[HookEvent] | None = None):
        if name:
            self.name = name
        if not self.name:
            self.name = self.__class__.__name__
        if events is not None:
            self.events = set(events)

    async def should_fire(self, context: HookContext) -> bool:  # pragma: no cover
        return True

    async def handle(self, context: HookContext) -> HookResult:
        raise NotImplementedError


class SemanticHook(Hook):
    """A hook that gates itself on a semantic predicate rather than a
    fixed lifecycle event.

    Subclasses implement ``evaluate(context) -> bool``. The framework
    calls ``evaluate`` and runs ``handle`` only when it returns True.
    """

    async def evaluate(self, context: HookContext) -> bool:
        return True

    async def should_fire(self, context: HookContext) -> bool:
        return await self.evaluate(context)


# ---------------------------------------------------------------------------
# Handler kinds (concrete adapters)
# ---------------------------------------------------------------------------


class PythonHook(Hook):
    """Wraps an arbitrary async callable as a hook."""

    def __init__(
        self,
        name: str,
        events: set[HookEvent],
        handler: Callable[[HookContext], Awaitable[HookResult | None]],
        condition: Callable[[HookContext], Awaitable[bool] | bool] | None = None,
    ):
        super().__init__(name=name, events=set(events))
        self._handler = handler
        self._condition = condition

    async def should_fire(self, context: HookContext) -> bool:
        if self._condition is None:
            return True
        result = self._condition(context)
        if asyncio.iscoroutine(result):
            result = await result
        return bool(result)

    async def handle(self, context: HookContext) -> HookResult:
        result = await self._handler(context)
        if result is None:
            return HookResult()
        return result


class LLMHook(Hook):
    """A hook that consults an LLM and translates the response to a HookResult.

    ``parse_response`` extracts the action / answer / rewrite payload from
    the model's text. The default expects the model to return JSON; subclass
    to handle other formats.
    """

    def __init__(
        self,
        name: str,
        events: set[HookEvent],
        prompt_template: str,
        model_attr: str = "light_model",
    ):
        super().__init__(name=name, events=set(events))
        self.prompt_template = prompt_template
        self.model_attr = model_attr

    def render_prompt(self, context: HookContext) -> str:
        try:
            return self.prompt_template.format(**context.__dict__)
        except KeyError as exc:
            logger.warning("LLMHook %s prompt template missing key %s", self.name, exc)
            return self.prompt_template

    def parse_response(self, raw: str) -> HookResult:
        """Extract a HookResult from the model's text response.

        Real LLMs love to wrap JSON in markdown fences (``` ```json ... ``` ```)
        or surround it with prose. We try three strategies before giving up:

        1. Parse the raw text directly.
        2. Strip markdown code fences (``​```json`` … ``​```​``)
           and parse the inside.
        3. Extract the first ``{ … }`` block and parse that.

        Total parse failure is logged at WARNING level so the user knows
        their LLMHook isn't taking effect (previously it was a completely
        silent no-op — dogfood Friction #B-4).
        """
        candidates: list[str] = [raw]

        stripped = raw.strip()
        # Strategy 2: peel a fenced block if present.
        if stripped.startswith("```"):
            inner = stripped
            # drop opening fence (``` or ```json or ```yaml etc.)
            first_nl = inner.find("\n")
            if first_nl != -1:
                inner = inner[first_nl + 1:]
            # drop trailing fence
            if inner.rstrip().endswith("```"):
                inner = inner.rstrip()[:-3]
            candidates.append(inner.strip())

        # Strategy 3: first balanced-looking { ... }
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            candidates.append(raw[start:end + 1])

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return HookResult(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        logger.warning(
            "LLMHook %s could not parse model response as JSON; "
            "returning empty HookResult. Raw response preview: %r",
            self.name, raw[:200],
        )
        return HookResult(metadata={"raw": raw, "parse_failed": True})

    async def handle(self, context: HookContext) -> HookResult:
        agent = context.agent
        model = getattr(agent, self.model_attr, None)
        if model is None:
            logger.warning("LLMHook %s could not find agent.%s", self.name, self.model_attr)
            return HookResult()
        response = await model.chat(
            [{"role": "user", "content": self.render_prompt(context)}],
            temperature=0.2,
            max_tokens=200,
        )
        return self.parse_response(response.content)


class ShellHook(Hook):
    """A hook that shells out to an external command.

    The handler invokes ``command`` with the serialized HookContext on
    stdin (JSON) and parses its stdout as ``HookResult`` (JSON). Errors
    in the subprocess are logged and treated as a non-fatal continue.

    This is the integration point with Claude-Code-style external hook
    scripts and language-agnostic plugin authors.
    """

    def __init__(
        self,
        name: str,
        events: set[HookEvent],
        command: list[str],
        timeout_seconds: float = 30.0,
    ):
        super().__init__(name=name, events=set(events))
        self.command = command
        self.timeout_seconds = timeout_seconds

    async def handle(self, context: HookContext) -> HookResult:
        if not self.command or shutil.which(self.command[0]) is None:
            logger.warning("ShellHook %s: command %r not found", self.name, self.command)
            return HookResult()

        payload = {
            "event": context.event.value,
            "user_input": context.user_input,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "tool_name": context.tool_name,
            "tool_args": context.tool_args,
            "scenario": context.scenario,
            "react_iteration": context.react_iteration,
        }

        try:
            proc = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(json.dumps(payload).encode("utf-8")),
                timeout=self.timeout_seconds,
            )
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("ShellHook %s timed out after %.1fs", self.name, self.timeout_seconds)
            return HookResult()
        except OSError as exc:
            logger.warning("ShellHook %s failed to spawn: %s", self.name, exc)
            return HookResult()

        if proc.returncode != 0:
            logger.warning(
                "ShellHook %s exited with code %s; stderr=%s",
                self.name, proc.returncode, stderr_bytes.decode("utf-8", "replace")[:200],
            )
            return HookResult()

        stdout = stdout_bytes.decode("utf-8", "replace").strip()
        if not stdout:
            return HookResult()
        try:
            return HookResult(**json.loads(stdout))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("ShellHook %s produced invalid JSON: %s", self.name, exc)
            return HookResult(metadata={"raw": stdout})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class HookRegistry:
    """
    Holds the set of registered hooks indexed by event.

    Lookup is O(1) per event; dispatch is sequential per event but each
    hook's ``should_fire`` + ``handle`` is awaited individually so a
    handler that returns SHORT_CIRCUIT or ABORT stops the downstream
    handlers for that event.
    """

    def __init__(self) -> None:
        self._by_event: dict[HookEvent, list[Hook]] = {}

    def register(self, hook: Hook) -> None:
        if not hook.events:
            raise ValueError(
                f"Hook {hook.name!r} has no events declared. "
                f"Pass events=... to the constructor or set it on the class."
            )
        for event in hook.events:
            self._by_event.setdefault(event, []).append(hook)
        logger.debug("Registered hook %s for events %s", hook.name, [e.value for e in hook.events])

    def unregister(self, name: str) -> bool:
        removed = False
        for hooks in self._by_event.values():
            for h in list(hooks):
                if h.name == name:
                    hooks.remove(h)
                    removed = True
        return removed

    def list_hooks(self, event: HookEvent | None = None) -> list[str]:
        if event is None:
            return [h.name for hs in self._by_event.values() for h in hs]
        return [h.name for h in self._by_event.get(event, [])]

    async def dispatch(self, event: HookEvent, context: HookContext) -> HookResult:
        """
        Fire every hook subscribed to ``event``.

        Returns the *last* HookResult — or the SHORT_CIRCUIT/ABORT result
        the moment any hook produces one. Subsequent hooks are skipped
        when that happens.
        """
        result = HookResult()
        for hook in self._by_event.get(event, ()):
            try:
                if not await hook.should_fire(context):
                    continue
                hook_result = await hook.handle(context)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Hook %s for event %s raised %s — continuing dispatch.",
                    hook.name, event.value, type(exc).__name__,
                )
                continue
            if hook_result is None:
                hook_result = HookResult()
            result = hook_result
            if result.action in ("short_circuit", "abort"):
                logger.info("Hook %s requested %s on event %s",
                            hook.name, result.action, event.value)
                return result
        return result

    async def fire_semantic(self, event: HookEvent, context: HookContext) -> HookResult:
        """Alias for dispatch; the dichotomy is documentation only."""
        return await self.dispatch(event, context)
