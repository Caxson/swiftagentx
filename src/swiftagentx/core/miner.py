"""
Transcript mining (D5) — background loop stage 1.

Turns REPEATED ReAct tool sequences into Scenario candidates without an
extra LLM call. The Planner fast path (planner.py) already knows how to
turn one ``GeneratedPlan`` into a reviewable ``PlanStore`` candidate;
mining reuses that exact path — it just reconstructs the same templated
shape from what the ReAct loop *actually executed*, instead of what an
LLM proposed, then feeds it through ``PlanStore.add()`` unchanged. From
there a mined candidate is indistinguishable from a planner-generated
one: same reuse gate, same promotion gate, same reviewer surface.

Clustering key is the same tool + kwarg-name "shape" ``PlanStore`` already
dedupes candidates on — argument VALUES vary per request, only the
chain's STRUCTURE repeats. A shape only becomes a candidate once it has
recurred across ``min_cluster_size`` independent requests; a one-off tool
sequence is noise, not a pattern.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..tools.scenario import ToolChainStep
from .planner import GeneratedPlan, PlanStore, plan_shape, sanitize_intent


@dataclass
class ReactTranscript:
    """One completed ReAct run's executed tool-call sequence.

    ``baseline_output`` is the answer ReAct actually returned to the user
    for this request (direct tool output or synthesized final answer) —
    the ground truth D6's replay eval gate (``core/replay_eval.py``)
    compares a candidate chain's deterministic replay against.
    """

    user_input: str
    actions: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    baseline_output: str = ""


class TranscriptMiner:
    """Clusters ReAct transcripts by tool-chain shape into plan candidates."""

    def __init__(self, min_cluster_size: int = 3, max_steps: int = 5):
        self.min_cluster_size = max(1, min_cluster_size)
        self.max_steps = max(1, max_steps)

    def mine(
        self, transcripts: list[ReactTranscript], plan_store: PlanStore,
    ) -> list[str]:
        """Cluster ``transcripts`` and feed qualifying clusters into
        ``plan_store``. Returns the plan_ids touched this run — either
        newly created candidates or existing ones that gained a fresh
        source-query anchor."""
        clusters: dict[str, list[tuple[ReactTranscript, GeneratedPlan]]] = defaultdict(list)
        for t in transcripts:
            if not (2 <= len(t.actions) <= self.max_steps):
                continue  # not a chain, or beyond the fast path's scope
            plan = _plan_from_transcript(t)
            clusters[plan_shape(plan.steps)].append((t, plan))

        touched: list[str] = []
        for group in clusters.values():
            if len(group) < self.min_cluster_size:
                continue
            for t, plan in group:
                cached = plan_store.add(plan, t.user_input)
                if cached.plan_id not in touched:
                    touched.append(cached.plan_id)
        return touched


def _plan_from_transcript(t: ReactTranscript) -> GeneratedPlan:
    """Reconstruct a templated ``GeneratedPlan`` from one executed chain.

    Each kwarg becomes a ``$param_name`` slot bound to the value that was
    actually passed — the same templating shape ``Planner.parse`` produces
    from an LLM plan, so the result is a drop-in for ``PlanStore.add()``.
    """
    steps = [
        ToolChainStep(
            tool=tool_name,
            kwargs_template={str(k): f"${k}" for k in kwargs},
        )
        for tool_name, kwargs in t.actions
    ]
    slots = {str(k): str(v) for _, kwargs in t.actions for k, v in kwargs.items()}
    intent = sanitize_intent("mined_" + "_".join(name for name, _ in t.actions))
    return GeneratedPlan(
        intent=intent, description="mined from ReAct transcripts",
        steps=steps, slots=slots,
    )
