"""
Tool base classes and type definitions.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class ToolOutputType(Enum):
    """Tool output type."""
    LLM_PROCESSED = "llm_processed"
    DIRECT_OUTPUT = "direct_output"
    SCRIPT_OUTPUT = "script_output"


class ToolOutput(BaseModel):
    """Tool execution output."""
    success: bool
    result: Any
    error: str | None = None
    output_type: ToolOutputType = ToolOutputType.LLM_PROCESSED
    metadata: dict[str, Any] = {}

    model_config = {"use_enum_values": False}


@runtime_checkable
class AgentContext(Protocol):
    """Tool execution context protocol — users can provide custom implementations.

    Any object that has these attributes will work as a context for tool execution.
    The built-in SessionContext satisfies this protocol.
    """
    variables: dict[str, Any]
    user_input: str
    session_id: str


class Tool(ABC):
    """
    Base class for all tools.

    Subclass this and implement `execute` to create custom tools.
    """

    def __init__(
        self,
        name: str,
        description: str,
        category: str = "general",
        output_type: ToolOutputType = ToolOutputType.LLM_PROCESSED,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        self.name = name
        self.description = description
        self.category = category
        self.output_type = output_type
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        # Opt out of context offload: a tool that already bounds its own
        # output, or whose whole purpose is reading offloaded content back
        # into context, must not have that output offloaded again.
        self.offload_exempt = False

    @abstractmethod
    async def execute(self, context: AgentContext, **kwargs: Any) -> ToolOutput:
        """
        Execute the tool.

        Args:
            context: Agent execution context (satisfies AgentContext protocol)
            **kwargs: Tool input parameters

        Returns:
            ToolOutput with results
        """
        ...

    def validate_input(self, **kwargs: Any) -> bool:
        """Validate tool input (override for custom validation)."""
        return True

    def get_schema(self) -> dict[str, Any]:
        """Get tool JSON schema (for LLM function calling)."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "output_type": self.output_type.value,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
