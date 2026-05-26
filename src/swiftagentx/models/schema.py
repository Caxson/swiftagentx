"""
SwiftAgent core data models.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StreamEventType(Enum):
    """Stream event type enumeration."""
    # Initialization
    INITIALIZED = "initialized"
    KB_LOOKUP = "kb_lookup"
    CACHE_HIT = "cache_hit"

    # Execution phase
    THOUGHT_START = "thought_start"
    THOUGHT_CHUNK = "thought_chunk"
    THOUGHT_END = "thought_end"

    ACTION_START = "action_start"
    ACTION_CHUNK = "action_chunk"
    ACTION_END = "action_end"

    OBSERVATION_START = "observation_start"
    OBSERVATION = "observation"
    OBSERVATION_END = "observation_end"

    # Final phase
    ANSWER_START = "answer_start"
    ANSWER_CHUNK = "answer_chunk"
    ANSWER_END = "answer_end"
    MESSAGE_END = "message_end"

    # Error and completion
    ERROR = "error"
    COMPLETED = "completed"


class StreamEvent(BaseModel):
    """Stream event object."""
    event_type: StreamEventType
    data: dict[str, Any] = Field(default_factory=dict)
    step_index: int = 0
    iteration: int = 0
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

    def to_sse_message(self) -> str:
        """Convert to SSE message format."""
        if self.event_type == StreamEventType.ANSWER_CHUNK:
            event_data = {
                "event": "message",
                "answer": self.data.get("answer", ""),
            }
            for key in ("conversation_id", "message_id", "task_id"):
                if key in self.data:
                    event_data[key] = self.data[key]
            event_data["created_at"] = int(self.timestamp)
            return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        else:
            event_data = {
                "event": self.event_type.value,
                "step": self.step_index,
                "iteration": self.iteration,
                "data": self.data,
                "timestamp": self.timestamp,
            }
            return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"


class ContextParameters(BaseModel):
    """Session context parameters container."""
    app_version: str = "1.0.0"
    platform: str = "default"
    device_id: str | None = None
    channel: str = "default"
    extra_params: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if key in ("app_version", "platform", "device_id", "channel"):
            return getattr(self, key, default)
        return self.extra_params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if key in ("app_version", "platform", "device_id", "channel"):
            setattr(self, key, value)
        else:
            self.extra_params[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_version": self.app_version,
            "platform": self.platform,
            "device_id": self.device_id,
            "channel": self.channel,
            **self.extra_params,
        }


class SessionContext(BaseModel):
    """Session context for agent execution."""
    session_id: str
    user_id: str
    user_input: str
    parameters: ContextParameters | None = Field(default_factory=ContextParameters)

    # Execution state
    current_iteration: int = 0
    max_iterations: int = 10

    # Memory and step tracking
    steps: list[dict[str, Any]] = Field(default_factory=list)
    variables: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"arbitrary_types_allowed": True}

    def add_step(
        self,
        step_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        input_data: Any = None,
        output_data: Any = None,
    ) -> None:
        step = {
            "type": step_type,
            "content": content,
            "iteration": self.current_iteration,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "input": input_data,
            "output": output_data,
        }
        self.steps.append(step)
        self.updated_at = datetime.now()

    def set_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value
        self.updated_at = datetime.now()

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.messages.append(message)
        self.updated_at = datetime.now()


class AgentRequest(BaseModel):
    """Agent invocation request."""
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    user_input: str = Field(..., description="User input text")

    app_version: str = Field("1.0.0", description="App version")
    platform: str = Field("default", description="Platform")
    device_id: str | None = Field(None, description="Device ID")
    channel: str = Field("default", description="Channel")
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Custom parameters")


class AgentResponse(BaseModel):
    """Agent invocation response."""
    session_id: str
    request_id: str
    answer: str
    total_iterations: int
    execution_time_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionRequest(BaseModel):
    """Tool execution request."""
    tool_name: str = Field(..., description="Tool name")
    tool_input: dict[str, Any] = Field(default_factory=dict, description="Tool input parameters")
    session_id: str = Field(..., description="Session ID")
    iteration: int = Field(0, description="Current iteration")


class ToolExecutionResult(BaseModel):
    """Tool execution result."""
    tool_name: str
    success: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThoughtStep(BaseModel):
    """Thought step in ReAct loop."""
    iteration: int
    content: str
    reasoning: str
    confidence: float = 0.5
    timestamp: datetime = Field(default_factory=datetime.now)


class ActionStep(BaseModel):
    """Action step in ReAct loop."""
    iteration: int
    tool_name: str
    tool_input: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ObservationStep(BaseModel):
    """Observation step in ReAct loop."""
    iteration: int
    tool_result: Any
    interpretation: str
    timestamp: datetime = Field(default_factory=datetime.now)
