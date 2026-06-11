"""
SwiftAgent framework configuration model.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    """Model tier for dual-model strategy."""
    LIGHT = "light"
    HEAVY = "heavy"


class SwiftAgentConfig(BaseModel):
    """Framework-level configuration."""

    # Agent identity
    name: str = "SwiftAgent"
    description: str = "An intelligent agent powered by SwiftAgent framework"
    system_prompt: str = ""

    # ReAct loop
    max_iterations: int = 10
    max_retries: int = 3

    # Cache
    enable_cache: bool = True
    kb_cache_ttl: int = 3600
    code_cache_ttl: int = 300

    # Input validation
    max_input_length: int = 10000
    debug: bool = False

    # Streaming
    sse_heartbeat_interval: float = 5.0
    sse_timeout: int = 120

    # Cache limits (0 = unlimited)
    max_cache_entries_per_level: int = 10000

    # Logging
    enable_request_tracing: bool = True
    log_level: str = "INFO"

    # Knowledge base
    kb_exact_match_threshold: float = 0.95

    # Scenario retrieval pre-filter: max candidates shown to the intent
    # classifier per request. Pools at or below this size are passed through
    # unfiltered.
    scenario_prefilter_top_k: int = 8

    # Layered memory (v0.3)
    memory_l2_size: int = 4
    memory_l3_max_size: int = 6
    memory_summarize_every_n_turns: int = 5
    memory_summarize_in_background: bool = True
    memory_enable_topic_change_hook: bool = True

    # Custom settings (extensible)
    extra: dict[str, Any] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key, default)
