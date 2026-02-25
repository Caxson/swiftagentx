"""
Shared test fixtures for SwiftAgent tests.
"""

import pytest
from swiftagentx import Agent, DummyModelClient, Tool, ToolOutput


class MockTool(Tool):
    """A mock tool for testing."""
    def __init__(self, name: str = "mock_tool", result: str = "mock result"):
        super().__init__(name=name, description=f"Mock tool: {name}")
        self._result = result

    async def execute(self, context, **kwargs):
        return ToolOutput(success=True, result=self._result)


@pytest.fixture
def dummy_model():
    return DummyModelClient(api_key="test-key", model="dummy-model")


@pytest.fixture
def basic_agent(dummy_model):
    return Agent(name="TestAgent", model=dummy_model, max_iterations=3)
