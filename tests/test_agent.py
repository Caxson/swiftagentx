"""
Tests for the core Agent class.
"""


import pytest

from swiftagentx import (
    Agent,
    AgentRequest,
    DummyModelClient,
    SSEStreamAdapter,
    Tool,
    ToolOutput,
)


class EchoTool(Tool):
    def __init__(self):
        super().__init__(name="echo", description="Echoes input")

    async def execute(self, context, **kwargs):
        text = kwargs.get("text", "echoed")
        return ToolOutput(success=True, result=f"Echo: {text}")


@pytest.fixture
def agent():
    return Agent(
        name="TestAgent",
        model=DummyModelClient(api_key="test", model="dummy"),
        max_iterations=3,
    )


@pytest.fixture
def agent_with_tool(agent):
    agent.register_tool(EchoTool())
    return agent


class TestAgentBasic:
    @pytest.mark.asyncio
    async def test_agent_run_returns_response(self, agent):
        response = await agent.run("Hello")
        assert response.answer is not None
        assert response.session_id is not None
        assert response.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_agent_run_with_context(self, agent):
        response = await agent.run("Hi", user_id="user1", session_id="sess1")
        assert response.session_id == "sess1"

    @pytest.mark.asyncio
    async def test_agent_with_tool(self, agent_with_tool):
        response = await agent_with_tool.run("echo hello")
        assert response.answer is not None

    @pytest.mark.asyncio
    async def test_agent_stream(self, agent):
        request = AgentRequest(
            user_id="user1",
            session_id="sess1",
            user_input="Hello stream",
        )
        adapter = SSEStreamAdapter()
        response = await agent.run_stream(request, adapter)
        assert response.answer is not None
        assert adapter.is_finished


class TestAgentSecurity:
    @pytest.mark.asyncio
    async def test_input_length_validation(self):
        """Over-length input is rejected as an AgentResponse — does NOT
        leak a ValueError out of run() (which would otherwise 500 a
        web handler that forgot try/except)."""
        from swiftagentx.models.config import SwiftAgentConfig
        agent = Agent(
            model=DummyModelClient(api_key="test", model="dummy"),
            config=SwiftAgentConfig(max_input_length=10),
        )
        response = await agent.run("a" * 100)
        assert response.metadata.get("input_rejected") is True
        assert response.metadata.get("error_class") == "ValueError"
        assert "exceeds maximum length" in response.answer

    @pytest.mark.asyncio
    async def test_error_sanitized_in_production(self):
        """In non-debug mode, error details should not leak to user."""
        from swiftagentx.models.config import SwiftAgentConfig

        class BrokenAgent(Agent):
            async def on_request_start(self, context):
                raise RuntimeError("secret database password: p@ss123")

        agent = BrokenAgent(
            model=DummyModelClient(api_key="test", model="dummy"),
            config=SwiftAgentConfig(debug=False),
        )
        response = await agent.run("hello")
        assert "p@ss123" not in response.answer
        assert "internal error" in response.answer.lower()
        assert "error" not in response.metadata  # no error details in metadata

    @pytest.mark.asyncio
    async def test_error_exposed_in_debug_mode(self):
        from swiftagentx.models.config import SwiftAgentConfig

        class BrokenAgent(Agent):
            async def on_request_start(self, context):
                raise RuntimeError("debug info here")

        agent = BrokenAgent(
            model=DummyModelClient(api_key="test", model="dummy"),
            config=SwiftAgentConfig(debug=True),
        )
        response = await agent.run("hello")
        assert "debug info here" in response.answer
        assert "error" in response.metadata

    @pytest.mark.asyncio
    async def test_stream_input_validation(self):
        """run_stream also returns an AgentResponse rather than raising."""
        from swiftagentx.models.config import SwiftAgentConfig
        agent = Agent(
            model=DummyModelClient(api_key="test", model="dummy"),
            config=SwiftAgentConfig(max_input_length=5),
        )
        request = AgentRequest(
            user_id="u1", session_id="s1", user_input="a" * 100,
        )
        adapter = SSEStreamAdapter()
        response = await agent.run_stream(request, adapter)
        assert response.metadata.get("input_rejected") is True
        assert "exceeds maximum length" in response.answer


class TestAgentConfig:
    def test_agent_name(self, agent):
        assert agent.name == "TestAgent"

    def test_agent_max_iterations(self, agent):
        assert agent.max_iterations == 3

    def test_agent_register_tool(self, agent):
        agent.register_tool(EchoTool())
        assert "echo" in agent.tool_registry.list_tools()

    def test_agent_model_access(self, agent):
        model = agent.heavy_model
        assert model is not None
        assert model.model == "dummy"
