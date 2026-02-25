"""
Tests for the tool system.
"""

import pytest
from swiftagentx import (
    Tool, ToolOutput, ToolOutputType,
    ToolRegistry, ToolExecutor,
    SessionContext,
)


class MockTool(Tool):
    def __init__(self, name="mock_tool"):
        super().__init__(name=name, description="A mock tool")

    async def execute(self, context, **kwargs):
        return ToolOutput(success=True, result={"message": f"mock result for {kwargs}"})


class FailingTool(Tool):
    def __init__(self):
        super().__init__(name="fail_tool", description="Always fails", max_retries=1)

    async def execute(self, context, **kwargs):
        raise ValueError("Tool failure")


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        assert registry.get("mock_tool") is tool

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockTool())

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        assert registry.unregister("mock_tool")
        assert registry.get("mock_tool") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(MockTool("tool_a"))
        registry.register(MockTool("tool_b"))
        tools = registry.list_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools

    def test_get_schemas(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        schemas = registry.get_schemas()
        assert "mock_tool" in schemas
        assert schemas["mock_tool"]["name"] == "mock_tool"


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        registry = ToolRegistry()
        registry.register(MockTool())
        executor = ToolExecutor(registry)
        context = SessionContext(session_id="s1", user_id="u1", user_input="test")
        result = await executor.execute("mock_tool", context, param1="val1")
        assert result.success
        assert "message" in result.result

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        context = SessionContext(session_id="s1", user_id="u1", user_input="test")
        result = await executor.execute("nonexistent", context)
        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_failure_with_retry(self):
        registry = ToolRegistry()
        registry.register(FailingTool())
        executor = ToolExecutor(registry)
        context = SessionContext(session_id="s1", user_id="u1", user_input="test")
        result = await executor.execute("fail_tool", context)
        assert not result.success
        assert "failed after" in result.error


class TestToolOutput:
    def test_tool_output_default_type(self):
        output = ToolOutput(success=True, result="data")
        assert output.output_type == ToolOutputType.LLM_PROCESSED

    def test_tool_output_direct(self):
        output = ToolOutput(success=True, result="data", output_type=ToolOutputType.DIRECT_OUTPUT)
        assert output.output_type == ToolOutputType.DIRECT_OUTPUT
