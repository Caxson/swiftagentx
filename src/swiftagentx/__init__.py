"""
SwiftAgent — Enterprise-grade fast-response Agent framework.

Features:
- Dual-model strategy (light for classification, heavy for execution)
- Scenario toolchains (skip ReAct for high-frequency patterns)
- Three-level cache (KB / tool result / session)
- SSE streaming with fine-grained events
- Production-ready (middleware, tracing, exponential backoff)
"""

from .admin.service import AdminService
from .core.agent import Agent
from .core.cache import CacheManager
from .core.memory import Message, SessionMemory
from .core.hooks import (
    Hook,
    HookContext,
    HookEvent,
    HookRegistry,
    HookResult,
    LLMHook,
    PythonHook,
    SemanticHook,
    ShellHook,
)
from .core.memory_hooks import TopicChangeHook
from .core.memory_layers import (
    DialogTurn,
    InMemoryBackend,
    LayeredMemory,
    LayeredMemoryConfig,
    LayeredMemoryStore,
    MemoryBackend,
    MemorySnapshot,
)
from .core.prompt_layout import PromptLayout
from .core.skills import Skill, SkillRegistry, parse_skill_markdown
from .core.subagent import (
    SubAgentInvocation,
    SubAgentManager,
    SubAgentRequest,
    SubAgentResult,
    SubAgentRole,
)
from .core.workspace import (
    InMemoryWorkspaceBackend,
    LocalDiskWorkspaceBackend,
    Workspace,
    WorkspaceBackend,
    use_workspace,
)
from .providers.mcp import MCPClient, MCPClientError, MCPServerSpec, MCPTool
from .core.model_client import DummyModelClient, ModelClient, ModelClientFactory, ModelResponse
from .core.pipeline import PipelineStage, RequestPipeline, StageAction, StageResult
from .core.prompt import PromptManager, PromptTemplate
from .core.router import IntentLevel, IntentResult, IntentRouter
from .knowledge_base.base import KnowledgeBase
from .knowledge_base.document import Document, SearchResult
from .knowledge_base.memory import MemoryKnowledgeBase
from .knowledge_base.stage import KnowledgeBaseStage
from .knowledge_base.tool import KnowledgeBaseTool
from .middleware.base import Middleware, MiddlewareChain
from .models.config import ModelTier, SwiftAgentConfig
from .models.schema import AgentRequest, AgentResponse, SessionContext
from .storage.base import StorageBackend
from .storage.memory import MemoryStorage
from .stream.adapter import SSEStreamAdapter
from .stream.builder import SSEEventBuilder
from .tools.base import AgentContext, Tool, ToolOutput, ToolOutputType
from .tools.executor import ToolExecutor
from .tools.registry import ToolRegistry
from .tools.scenario import ScenarioConfig, ScenarioEngine, ToolChainStep

__version__ = "0.3.3"

__all__ = [
    # Core
    "Agent",
    "ModelClient", "ModelResponse", "DummyModelClient", "ModelClientFactory",
    "Message", "SessionMemory",
    # v0.3 layered memory
    "DialogTurn", "MemorySnapshot",
    "MemoryBackend", "InMemoryBackend",
    "LayeredMemoryConfig", "LayeredMemory", "LayeredMemoryStore",
    # v0.3 hooks
    "Hook", "SemanticHook", "PythonHook", "LLMHook", "ShellHook",
    "HookEvent", "HookContext", "HookResult", "HookRegistry",
    "TopicChangeHook",
    # v0.3 MCP
    "MCPClient", "MCPClientError", "MCPServerSpec", "MCPTool",
    # v0.3 sub-agents
    "SubAgentRole", "SubAgentRequest", "SubAgentResult",
    "SubAgentManager", "SubAgentInvocation",
    # v0.3 skills
    "Skill", "SkillRegistry", "parse_skill_markdown",
    # v0.3 workspace
    "Workspace", "WorkspaceBackend",
    "InMemoryWorkspaceBackend", "LocalDiskWorkspaceBackend",
    "use_workspace",
    # v0.3 prompt layout
    "PromptLayout",
    "CacheManager",
    "PromptManager", "PromptTemplate",
    "IntentLevel", "IntentResult", "IntentRouter",
    "PipelineStage", "StageResult", "StageAction", "RequestPipeline",
    # Tools
    "Tool", "ToolOutput", "ToolOutputType", "AgentContext",
    "ToolRegistry", "ToolExecutor",
    "ScenarioConfig", "ToolChainStep", "ScenarioEngine",
    # Stream
    "SSEStreamAdapter", "SSEEventBuilder",
    # Models
    "AgentRequest", "AgentResponse", "SessionContext",
    "SwiftAgentConfig", "ModelTier",
    # Middleware
    "Middleware", "MiddlewareChain",
    # Storage
    "StorageBackend", "MemoryStorage",
    # Knowledge Base
    "Document", "SearchResult",
    "KnowledgeBase", "MemoryKnowledgeBase",
    "KnowledgeBaseTool", "KnowledgeBaseStage",
    # Admin
    "AdminService",
    # Version
    "__version__",
]
