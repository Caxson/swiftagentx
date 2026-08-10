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

    # Planner fast path (opt-in). Before entering the ReAct loop, spend ONE
    # light-model call generating a deterministic tool plan; any failure
    # falls back to ReAct. Caveat: if a plan fails mid-chain after a
    # side-effectful tool already ran, the ReAct fallback may run that tool
    # again — enable with care for non-idempotent toolsets.
    enable_planner: bool = False
    # A successful plan's afterlife has two gates, each defaulting to
    # manual (nothing persists behind the developer's back):
    #
    # Gate 1 — REUSE. plan_auto_reuse=False (default): generated plans are
    # one-shot; they accumulate as reviewable candidates (same-shape
    # regenerations dedupe and keep score) until Agent.approve_plan()
    # opens the gate. True: a plan that executed successfully is matched
    # and reused for later requests immediately.
    plan_auto_reuse: bool = False
    # Gate 2 — PROMOTION to Scenario. plan_auto_promote=True auto-registers
    # a plan as a Scenario after `plan_promote_after` successes with zero
    # failures; default False keeps it manual
    # (Agent.promote_plan / export_plan_scenario).
    plan_auto_promote: bool = False
    plan_promote_after: int = 3

    # Transcript mining (D5, opt-in). When on, every successful ReAct run
    # with a 2+ tool chain is logged in-memory; Agent.mine_scenario_candidates()
    # (called by an operator-driven background task — this framework has no
    # built-in scheduler) clusters the log by tool-chain shape and feeds
    # clusters that recur at least `mining_min_cluster_size` times into the
    # same plan_store candidate pool the Planner fast path uses. Mined
    # candidates land unapproved/unpromoted — same manual review gates as
    # any other plan candidate.
    enable_transcript_mining: bool = False
    mining_min_cluster_size: int = 3
    mining_max_transcripts: int = 500

    # Replay eval gate (D6, opt-in). When on, every transcript recorded for
    # mining is also kept in a rolling eval log; Agent.replay_eval_plan()
    # (operator-driven, same as mine_scenario_candidates()) replays a
    # candidate's tool chain against its matching historical requests and
    # scores agreement against what ReAct actually answered. A candidate
    # that clears `eval_pass_rate_threshold` is auto-approved into the
    # reuse queue (PlanStore.approve); otherwise it stays an unapproved
    # candidate awaiting manual review, same as today.
    enable_replay_eval: bool = False
    eval_max_transcripts: int = 200
    eval_min_cases: int = 1
    eval_agreement_threshold: float = 0.5
    eval_pass_rate_threshold: float = 0.7

    # Context offload (v0.5): tool results whose stringified form exceeds
    # this many characters are written to the session workspace instead of
    # being inlined into the ReAct / Scenario context sent to the LLM; the
    # context keeps a preview + a workspace file reference that the
    # `workspace_read` tool can pull back on demand. 0 disables offloading.
    context_offload_threshold: int = 4000
    context_offload_preview_chars: int = 500
    # Per-call ceiling for `workspace_read`. Its output is exempt from
    # offload (otherwise reading an offloaded result back would just offload
    # it again), so this is what keeps a read-back bounded; the model pages
    # through a larger file with the `offset` parameter.
    context_offload_read_chunk_chars: int = 4000

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
