"""
Scenario toolchain engine — executes pre-defined tool chains for high-frequency scenarios.

Scenarios skip the full ReAct loop, saving 2-3 LLM calls for common request patterns.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from string import Template
from typing import Any

from pydantic import BaseModel, Field

from ..core.workspace import Workspace
from .base import AgentContext, ToolOutput, ToolOutputType
from .executor import ToolExecutor

logger = logging.getLogger(__name__)


class ToolChainStep(BaseModel):
    """A single step in a scenario tool chain.

    Two ways to populate the tool's kwargs from the scenario's variable bag:

    - ``query_template``: a single ``$foo`` Template string. The result is
      passed as the ``query`` kwarg. Legacy / single-arg tools.
    - ``kwargs_template``: a dict of ``{kwarg_name: $foo Template}``. Used
      by multi-parameter tools — including MCP tools whose schema is
      ``add(a, b)``-shaped rather than ``tool(query=...)``-shaped.

    Pick one. ``kwargs_template`` wins if both are set.
    """
    tool: str
    extract_to: str = ""
    condition: str = "always"
    query_template: str = ""
    kwargs_template: dict[str, str] = Field(default_factory=dict)


# A tool_chain entry is either a single step, or a *parallel group*: a list
# of steps with no dependencies between them, fanned out with
# ``asyncio.gather`` and joined before the chain continues. Kept as a plain
# list-of-lists on purpose — no graph/DAG abstraction.
ToolChainEntry = ToolChainStep | list[ToolChainStep]


class ScenarioConfig(BaseModel):
    """Configuration for a scenario toolchain."""
    name: str
    description: str = ""
    triggers: list[str] = Field(default_factory=list)
    tool_chain: list[ToolChainEntry] = Field(default_factory=list)
    cache_key_template: str = ""
    cache_ttl: int = 3600
    output_template: str = "llm"
    output_type: str = "llm_processed"  # "direct" | "llm_processed"
    on_group_failure: str = "fail_fast"  # "fail_fast" | "best_effort"

    def iter_steps(self) -> list[ToolChainStep]:
        """Flatten ``tool_chain`` into its individual steps, in order —
        parallel groups expand in place."""
        steps: list[ToolChainStep] = []
        for entry in self.tool_chain:
            steps.extend(entry if isinstance(entry, list) else [entry])
        return steps

    def required_vars(self) -> set[str]:
        """Template variables this scenario must be given from outside.

        Scans every step's ``query_template`` / ``kwargs_template`` for
        ``$name`` tokens, then subtracts (a) reserved keys always supplied
        by the agent and (b) any var produced by an earlier step's
        ``extract_to`` (those are filled mid-chain, not by the user).

        The agent uses this to know which slots to extract from natural
        language so a Scenario like ``weather(city=$city)`` actually fires
        when the user just types "北京天气怎么样" — without the caller
        having to pre-parse ``city`` and pass it as a kwarg.
        """
        import re

        reserved = {"user_input", "user_id", "session_id"}
        steps = self.iter_steps()
        produced = {s.extract_to for s in steps if s.extract_to}
        names: set[str] = set()
        for step in steps:
            templates = list(step.kwargs_template.values())
            if step.query_template:
                templates.append(step.query_template)
            if step.condition:
                templates.append(step.condition)
            for tmpl in templates:
                names.update(re.findall(r"\$(\w+)", tmpl))
        return names - reserved - produced


class ScenarioCheckpoint:
    """Persists a scenario chain's progress after every completed step-group
    so a chain interrupted mid-run (process crash, or paused for a human
    approval step upstream of `Agent.promote_plan`) can resume without
    re-running steps that already succeeded.

    Deliberately minimal (docs/OPTIMIZATION_PLAN.md D4): one JSON blob per
    ``(session, key)``, written through the same ``Workspace`` abstraction
    D3 already uses for context offload — no new storage backend, no graph/
    state-machine class. Unlike D3's offload keys, the checkpoint path is
    *not* suffixed with a random id: it must be the same path every time so
    a fresh process can find and resume it.
    """

    def __init__(self, workspace: Workspace, key: str) -> None:
        self._workspace = workspace
        self._relative = f"checkpoints/{key}.json"

    async def load(self) -> dict[str, Any] | None:
        """Return the last saved ``{group_index, collected, failed_steps}``,
        or ``None`` if this chain has never been checkpointed (or already
        ran to completion and was cleared)."""
        raw = await self._workspace.read(self._relative)
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))

    async def save(
        self, *, group_index: int, collected: dict[str, Any], failed_steps: list[str],
    ) -> None:
        payload = {"group_index": group_index, "collected": collected, "failed_steps": failed_steps}
        await self._workspace.write(self._relative, json.dumps(payload))

    async def clear(self) -> None:
        """Drop the checkpoint once the chain has run to completion — a
        later invocation of the same scenario/session must start fresh,
        not resume a finished run."""
        await self._workspace.remove(self._relative)


class ScenarioEngine:
    """
    Scenario toolchain execution engine.

    Register scenarios with tool chains, match user input to scenarios,
    and execute the chain without going through the full ReAct loop.
    """

    def __init__(self) -> None:
        self._scenarios: dict[str, ScenarioConfig] = {}
        self._cache_key_builders: dict[str, Callable[..., str]] = {}

    def register(self, scenario_id: str, scenario: ScenarioConfig) -> None:
        self._scenarios[scenario_id] = scenario
        logger.info(f"Registered scenario: {scenario_id} ({scenario.name})")

    def register_cache_key_builder(self, scenario_id: str, builder: Callable[..., str]) -> None:
        """Register a custom cache key builder for a scenario."""
        self._cache_key_builders[scenario_id] = builder

    def unregister(self, scenario_id: str) -> None:
        self._scenarios.pop(scenario_id, None)
        self._cache_key_builders.pop(scenario_id, None)

    def get(self, scenario_id: str) -> ScenarioConfig | None:
        return self._scenarios.get(scenario_id)

    def list_scenarios(self) -> dict[str, dict[str, Any]]:
        return {
            sid: {
                "name": sc.name,
                "description": sc.description,
                "triggers": sc.triggers,
                "cache_ttl": sc.cache_ttl,
                "tool_count": len(sc.iter_steps()),
            }
            for sid, sc in self._scenarios.items()
        }

    def match_by_id(self, scenario_id: str) -> ScenarioConfig | None:
        """Match scenario by explicit ID (used when LLM classifies intent)."""
        return self._scenarios.get(scenario_id)

    def is_cacheable(self, scenario_id: str) -> bool:
        """Whether this scenario opted into result caching.

        Caching is opt-in: only scenarios that declared a
        ``cache_key_template`` (or registered a custom key builder) are
        cached. A scenario that set neither is never cached, even though
        ``cache_ttl`` has a non-zero default — otherwise we'd silently
        cache scenarios the author never intended to.
        """
        if scenario_id in self._cache_key_builders:
            return True
        scenario = self._scenarios.get(scenario_id)
        return bool(scenario and scenario.cache_key_template)

    def build_cache_key(self, scenario_id: str, context: dict[str, Any]) -> str:
        """Build cache key using custom builder or default template."""
        if scenario_id in self._cache_key_builders:
            return self._cache_key_builders[scenario_id](scenario_id, context)

        scenario = self._scenarios.get(scenario_id)
        if not scenario or not scenario.cache_key_template:
            return f"{scenario_id}_{context.get('user_id', 'unknown')}"

        try:
            return Template(scenario.cache_key_template).safe_substitute(context)
        except (KeyError, ValueError):
            return f"{scenario_id}_{context.get('user_id', 'unknown')}"

    async def execute(
        self,
        scenario_id: str,
        context: AgentContext,
        tool_executor: ToolExecutor,
        extra_vars: dict[str, Any] | None = None,
        step_callback: Callable[..., Any] | None = None,
        checkpoint: ScenarioCheckpoint | None = None,
    ) -> ToolOutput:
        """
        Execute a scenario's tool chain.

        Args:
            scenario_id: Scenario identifier
            context: Agent execution context
            tool_executor: Tool executor instance
            extra_vars: Additional variables for query template rendering
            step_callback: Optional async callback invoked before and after
                each step. Signature: ``async def cb(phase: str, step: ToolChainStep,
                tool_kwargs: dict, output: ToolOutput | None) -> None``.
                ``phase`` is ``"before"`` or ``"after"``; ``output`` is
                ``None`` for the "before" call and the step's result for
                "after". Used by ``Agent._execute_scenario`` to dispatch
                ``HookEvent.BEFORE_SCENARIO_STEP`` / ``AFTER_SCENARIO_STEP``.
            checkpoint: Optional D4 checkpoint. When given, progress is
                persisted after every step-group and a prior unfinished run
                is resumed instead of restarted from scratch.

        Returns:
            Combined ToolOutput from the chain
        """
        scenario = self._scenarios.get(scenario_id)
        if not scenario:
            return ToolOutput(success=False, result=None, error=f"Scenario '{scenario_id}' not found")
        return await self.execute_config(
            scenario, scenario_id, context, tool_executor,
            extra_vars=extra_vars, step_callback=step_callback, checkpoint=checkpoint,
        )

    async def execute_config(
        self,
        scenario: ScenarioConfig,
        scenario_id: str,
        context: AgentContext,
        tool_executor: ToolExecutor,
        extra_vars: dict[str, Any] | None = None,
        step_callback: Callable[..., Any] | None = None,
        checkpoint: ScenarioCheckpoint | None = None,
    ) -> ToolOutput:
        """Execute a ScenarioConfig that need not be registered.

        Same semantics as :meth:`execute`, but takes the config object
        directly. This is what lets the Planner fast path run an ephemeral,
        LLM-generated plan through the exact same toolchain machinery as a
        registered Scenario — one execution engine, two front doors.
        """
        if not scenario.tool_chain:
            return ToolOutput(
                success=True,
                result={"scenario": scenario_id, "message": "No tool chain configured"},
                output_type=ToolOutputType.DIRECT_OUTPUT,
            )

        collected: dict[str, Any] = extra_vars or {}
        last_output: ToolOutput | None = None
        failed_steps: list[str] = []
        start_index = 0

        # D4: resume from a prior interrupted run. A saved checkpoint's
        # `collected` wins over `extra_vars` on key clashes — it reflects
        # everything the chain had already produced (including possibly a
        # newer value for a var also present in extra_vars) up to the point
        # it stopped.
        if checkpoint is not None:
            saved = await checkpoint.load()
            if saved is not None:
                start_index = saved["group_index"]
                collected = {**collected, **saved["collected"]}
                failed_steps = list(saved["failed_steps"])

        chain_completed = False
        for index, entry in enumerate(scenario.tool_chain):
            if index < start_index:
                continue

            # A plain step is just a parallel "group" of one — same code
            # path either way, no dependencies to reason about within a
            # group by construction. Conditions are evaluated here, against
            # `collected` as joined by the previous entry, so a step can
            # branch on an earlier group's results.
            group = [s for s in (entry if isinstance(entry, list) else [entry])
                     if self._eval_condition(s.condition, collected)]
            if not group:
                if checkpoint is not None:
                    await checkpoint.save(
                        group_index=index + 1, collected=collected, failed_steps=failed_steps,
                    )
                continue

            # Build every step's tool input from one snapshot of `collected`
            # taken before the group runs — steps in a group are declared
            # to have no dependencies on each other, so none of them may
            # see another group member's output.
            pairs: list[tuple[ToolChainStep, dict[str, Any]]] = [
                (step, self._render_kwargs(step, collected)) for step in group
            ]

            if step_callback is not None:
                for step, tool_kwargs in pairs:
                    await step_callback("before", step, tool_kwargs, None)

            outputs = await asyncio.gather(
                *(tool_executor.execute(step.tool, context, **tool_kwargs)
                  for step, tool_kwargs in pairs),
                return_exceptions=True,
            )

            group_failed = False
            for (step, tool_kwargs), output in zip(pairs, outputs, strict=True):
                if isinstance(output, BaseException):
                    output = ToolOutput(success=False, result=None, error=str(output))

                if output.success and step.extract_to:
                    collected[step.extract_to] = output.result

                last_output = output

                if step_callback is not None:
                    await step_callback("after", step, tool_kwargs, output)

                if not output.success:
                    logger.warning(f"Scenario '{scenario_id}' step '{step.tool}' failed: {output.error}")
                    group_failed = True
                    failed_steps.append(step.tool)

            # Retry the same (failed) group on the next resume; a
            # successful or best-effort group advances past it.
            if checkpoint is not None:
                await checkpoint.save(
                    group_index=index if group_failed else index + 1,
                    collected=collected, failed_steps=failed_steps,
                )

            if group_failed and scenario.on_group_failure != "best_effort":
                break
        else:
            chain_completed = True

        # The chain ran to completion (possibly with best-effort failures
        # along the way, but nothing left to resume) — a checkpoint from
        # this or an earlier interrupted run no longer applies.
        if checkpoint is not None and chain_completed:
            await checkpoint.clear()

        if last_output is None:
            return ToolOutput(success=True, result=collected, output_type=ToolOutputType.LLM_PROCESSED)

        # Determine output type + the value to return.
        #
        # For "direct" output: the user wants the raw tool result, period.
        # For "llm_processed": if any step declared extract_to we hand the
        # downstream LLM the structured ``collected`` dict; otherwise we
        # still hand it the last tool's result. The previous logic ("use
        # collected if truthy") returned the *initial* extra_vars dict
        # (user_id / user_input / session_id) when no step used extract_to
        # — that's the original Scenario-shortcut bug surfaced in
        # dogfooding (Friction #7).
        steps = scenario.iter_steps()
        any_extracted = any(step.extract_to for step in steps)
        if scenario.output_type == "direct":
            output_type = ToolOutputType.DIRECT_OUTPUT
            result_value: Any = last_output.result
        else:
            output_type = ToolOutputType.LLM_PROCESSED
            result_value = collected if any_extracted else last_output.result

        return ToolOutput(
            success=last_output.success,
            result=result_value,
            error=last_output.error,
            output_type=output_type,
            metadata={"scenario": scenario_id, "steps_executed": len(steps), "failed_steps": failed_steps},
        )

    @staticmethod
    def _eval_condition(condition: str, collected: dict[str, Any]) -> bool:
        """Evaluate a step's ``condition`` against the variable bag as
        joined by the previous entry — so a step can branch on an earlier
        (possibly parallel) group's results. Deliberately minimal, no
        expression language:

        - ``"always"`` (default) / ``""``: always run.
        - ``"never"``: never run.
        - ``"$var"``: run iff ``collected[var]`` is truthy.
        - ``"!$var"``: run iff ``collected[var]`` is falsy (or missing).
        - ``"$var == literal"`` / ``"$var != literal"``: string equality
          against the collected value.
        """
        condition = condition.strip()
        if condition in ("", "always"):
            return True
        if condition == "never":
            return False
        if condition.startswith("!$"):
            return not collected.get(condition[2:])
        if condition.startswith("$"):
            for op in ("==", "!="):
                if op in condition:
                    var_part, _, literal = condition.partition(op)
                    value = str(collected.get(var_part.strip().lstrip("$"), ""))
                    literal = literal.strip().strip("'\"")
                    return (value == literal) if op == "==" else (value != literal)
            return bool(collected.get(condition[1:]))
        return True

    @staticmethod
    def _render_kwargs(step: ToolChainStep, collected: dict[str, Any]) -> dict[str, Any]:
        """Render a step's ``kwargs_template``/``query_template`` against
        the variable bag collected so far.

        Prefers the dict-shaped ``kwargs_template`` (each value a ``$foo``
        Template) so multi-parameter tools — including most MCP tools whose
        schema is shaped like ``add(a, b)`` rather than ``tool(query)`` —
        work inside a Scenario chain. Falls back to the legacy single-
        "query" ``query_template`` for back compatibility.
        """
        tool_kwargs: dict[str, Any] = {}
        if step.kwargs_template:
            for key, tmpl in step.kwargs_template.items():
                try:
                    tool_kwargs[key] = Template(tmpl).safe_substitute(collected)
                except (KeyError, ValueError):
                    tool_kwargs[key] = tmpl
        elif step.query_template:
            try:
                tool_kwargs["query"] = Template(step.query_template).safe_substitute(collected)
            except (KeyError, ValueError):
                tool_kwargs["query"] = step.query_template
        return tool_kwargs
