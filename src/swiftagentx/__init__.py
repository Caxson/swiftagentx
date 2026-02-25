"""
SwiftAgent — Enterprise-grade fast-response Agent framework.

Features:
- Dual-model strategy (light for classification, heavy for execution)
- Scenario toolchains (skip ReAct for high-frequency patterns)
- Three-level cache (KB / tool result / session)
- SSE streaming with fine-grained events
- Production-ready (middleware, tracing, exponential backoff)
"""

from .core.agent import Agent
from .core.model_client import ModelClient, ModelResponse, DummyModelClient, ModelClientFactory
from .core.memory import Message, SessionMemory
from .core.cache import CacheManager
from .core.prompt import PromptManager, PromptTemplate
from .core.router import IntentLevel, IntentResult, IntentRouter
from .core.pipeline import PipelineStage, StageResult, RequestPipeline
from .tools.base import Tool, ToolOutput, ToolOutputType, AgentContext
from .tools.registry import ToolRegistry
from .tools.executor import ToolExecutor
from .tools.scenario import ScenarioConfig, ToolChainStep, ScenarioEngine
from .stream.adapter import SSEStreamAdapter
from .stream.builder import SSEEventBuilder
from .models.schema import AgentRequest, AgentResponse, SessionContext
from .models.config import SwiftAgentConfig, ModelTier
from .middleware.base import Middleware, MiddlewareChain
from .storage.base import StorageBackend
from .storage.memory import MemoryStorage
from .knowledge_base.document import Document, SearchResult
from .knowledge_base.base import KnowledgeBase
from .knowledge_base.memory import MemoryKnowledgeBase
from .knowledge_base.tool import KnowledgeBaseTool
from .knowledge_base.stage import KnowledgeBaseStage
from .admin.service import AdminService

__version__ = "0.1.0"

__all__ = [
    # Core
    "Agent",
    "ModelClient", "ModelResponse", "DummyModelClient", "ModelClientFactory",
    "Message", "SessionMemory",
    "CacheManager",
    "PromptManager", "PromptTemplate",
    "IntentLevel", "IntentResult", "IntentRouter",
    "PipelineStage", "StageResult", "RequestPipeline",
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
