"""
Tool executor — executes tools with retry, timeout, and exponential backoff.
"""

import asyncio
import json
from typing import Any, List, Optional
import logging
from .base import Tool, ToolOutput, AgentContext
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Tool executor with retry, timeout, and exponential backoff.
    """

    def __init__(self, registry: ToolRegistry, max_retries: int = 3):
        self.max_retries = max_retries
        self.registry = registry

    async def execute(
        self,
        tool_name: str,
        context: AgentContext,
        timeout_override: Optional[float] = None,
        max_retries_override: Optional[int] = None,
        **tool_input: Any,
    ) -> ToolOutput:
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolOutput(success=False, result=None, error=f"Tool '{tool_name}' not found in registry")

        if not tool.validate_input(**tool_input):
            return ToolOutput(success=False, result=None, error=f"Invalid input for tool '{tool_name}'")

        timeout = timeout_override or tool.timeout_seconds
        max_retries = max_retries_override if max_retries_override is not None else tool.max_retries

        logger.info(f"Tool call: {tool_name} | params: {json.dumps(tool_input, indent=2, ensure_ascii=False)}")

        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"  Attempt {attempt + 1}/{max_retries}...")
                result = await asyncio.wait_for(tool.execute(context, **tool_input), timeout=timeout)
                if result.success:
                    result_preview = str(result.result)[:200] if result.result else "None"
                    logger.info(f"  Tool '{tool_name}' succeeded (attempt {attempt + 1}): {result_preview}")
                else:
                    logger.warning(f"  Tool '{tool_name}' returned failure: {result.error}")
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout} seconds"
                logger.warning(f"Tool '{tool_name}' timed out on attempt {attempt + 1}")

            except Exception as e:
                last_error = str(e)
                logger.error(f"Tool '{tool_name}' failed on attempt {attempt + 1}: {last_error}", exc_info=True)

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff
                logger.info(f"Retrying tool '{tool_name}' after {wait_time}s...")
                await asyncio.sleep(wait_time)

        return ToolOutput(
            success=False, result=None,
            error=f"Tool '{tool_name}' failed after {max_retries} attempts: {last_error}",
        )

    async def execute_multiple(
        self, tools_input: List[dict], context: AgentContext, parallel: bool = False
    ) -> list:
        if parallel:
            tasks = [
                self.execute(
                    spec["tool_name"], context,
                    **{k: v for k, v in spec.items() if k != "tool_name"},
                )
                for spec in tools_input
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for spec in tools_input:
                tool_name = spec["tool_name"]
                params = {k: v for k, v in spec.items() if k != "tool_name"}
                result = await self.execute(tool_name, context, **params)
                results.append(result)
            return results

    def get_available_tools(self) -> List[str]:
        return self.registry.list_tools()

    def get_tool_schemas(self) -> dict:
        return self.registry.get_schemas()
