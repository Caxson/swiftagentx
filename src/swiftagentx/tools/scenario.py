"""
Scenario toolchain engine — executes pre-defined tool chains for high-frequency scenarios.

Scenarios skip the full ReAct loop, saving 2-3 LLM calls for common request patterns.
"""

import logging
from collections.abc import Callable
from string import Template
from typing import Any

from pydantic import BaseModel, Field

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


class ScenarioConfig(BaseModel):
    """Configuration for a scenario toolchain."""
    name: str
    description: str = ""
    triggers: list[str] = Field(default_factory=list)
    tool_chain: list[ToolChainStep] = Field(default_factory=list)
    cache_key_template: str = ""
    cache_ttl: int = 3600
    output_template: str = "llm"
    output_type: str = "llm_processed"  # "direct" | "llm_processed"


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
                "tool_count": len(sc.tool_chain),
            }
            for sid, sc in self._scenarios.items()
        }

    def match_by_id(self, scenario_id: str) -> ScenarioConfig | None:
        """Match scenario by explicit ID (used when LLM classifies intent)."""
        return self._scenarios.get(scenario_id)

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

        Returns:
            Combined ToolOutput from the chain
        """
        scenario = self._scenarios.get(scenario_id)
        if not scenario:
            return ToolOutput(success=False, result=None, error=f"Scenario '{scenario_id}' not found")

        if not scenario.tool_chain:
            return ToolOutput(
                success=True,
                result={"scenario": scenario_id, "message": "No tool chain configured"},
                output_type=ToolOutputType.DIRECT_OUTPUT,
            )

        collected: dict[str, Any] = extra_vars or {}
        last_output: ToolOutput | None = None

        for step in scenario.tool_chain:
            # Check condition
            if step.condition == "never":
                continue

            # Build tool input. Prefer the new dict-shaped kwargs_template
            # (each value is a $foo Template) so multi-parameter tools —
            # including most MCP tools whose schema is shaped like
            # add(a, b) rather than tool(query) — work inside a Scenario
            # chain. Fall back to the legacy single-"query" query_template
            # for back compatibility.
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

            if step_callback is not None:
                await step_callback("before", step, tool_kwargs, None)

            # Execute tool
            output = await tool_executor.execute(step.tool, context, **tool_kwargs)

            if output.success and step.extract_to:
                collected[step.extract_to] = output.result

            last_output = output

            if step_callback is not None:
                await step_callback("after", step, tool_kwargs, output)

            if not output.success:
                logger.warning(f"Scenario '{scenario_id}' step '{step.tool}' failed: {output.error}")
                break

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
        any_extracted = any(step.extract_to for step in scenario.tool_chain)
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
            metadata={"scenario": scenario_id, "steps_executed": len(scenario.tool_chain)},
        )
