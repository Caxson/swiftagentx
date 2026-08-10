"""
Replay eval gate (D6) — background loop stage 3.

D5 mines repeated ReAct tool chains into ``PlanStore`` candidates; this
module is the quality gate that runs *before* a candidate is trusted enough
to open its own reuse gate (``PlanStore.approve``). It replays a candidate's
tool chain — using the exact tool + kwargs pairs ReAct itself executed for
each historical request — and scores how closely the replayed tool output
agrees with the answer ReAct actually gave the user for that request. Only
a candidate whose agreement rate clears ``pass_rate_threshold`` across
enough historical cases gets auto-approved into the review/reuse queue;
everything else stays an unapproved candidate, same as today.

Deliberately reuses existing primitives instead of adding new abstractions:
``ToolExecutor`` runs the replay (same engine ReAct/Scenario/Planner all
share), ``retrieval.tokenize`` scores agreement (already used by
``PlanStore.match``), and ``Workspace`` persists the report (the same
storage seam D3/D4 use).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..tools.base import AgentContext, ToolOutput
from ..tools.executor import ToolExecutor
from ..tools.scenario import ToolChainStep
from .miner import ReactTranscript
from .planner import CachedPlan, plan_shape
from .retrieval import tokenize
from .workspace import Workspace


class ReplayCaseResult(BaseModel):
    """One historical request's replay outcome."""

    user_input: str
    agreement: float
    passed: bool
    replay_output: str
    baseline_output: str


class ReplayReport(BaseModel):
    """Aggregate replay-eval outcome for one candidate plan."""

    plan_id: str
    intent: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    agreement_threshold: float
    pass_rate_threshold: float
    verdict: bool
    cases: list[ReplayCaseResult] = Field(default_factory=list)


def agreement_score(replay_output: str, baseline_output: str) -> float:
    """Token-overlap (Jaccard) agreement between a replay and its baseline.

    Both empty counts as full agreement (nothing to disagree on); either
    side empty with the other non-empty is zero agreement.
    """
    replay_tokens = set(tokenize(replay_output))
    baseline_tokens = set(tokenize(baseline_output))
    if not replay_tokens and not baseline_tokens:
        return 1.0
    if not replay_tokens or not baseline_tokens:
        return 0.0
    return len(replay_tokens & baseline_tokens) / len(replay_tokens | baseline_tokens)


class ReplayEvaluator:
    """Replays candidate plans against historical ReAct transcripts.

    ``agreement_threshold`` is the per-case bar a replay's output must
    clear against that request's ReAct baseline to count as "passed".
    ``pass_rate_threshold`` is the fraction of cases that must pass for the
    candidate as a whole to clear the gate. Both default conservatively —
    a false-positive gate opens auto-reuse for a plan that doesn't actually
    agree with ReAct.
    """

    def __init__(
        self,
        tool_executor: ToolExecutor,
        agreement_threshold: float = 0.5,
        pass_rate_threshold: float = 0.7,
    ):
        self.tool_executor = tool_executor
        self.agreement_threshold = agreement_threshold
        self.pass_rate_threshold = pass_rate_threshold

    async def evaluate(
        self, plan: CachedPlan, cases: list[ReactTranscript], context: AgentContext,
    ) -> ReplayReport:
        """Replay every case's recorded tool chain and score it against
        that case's ReAct baseline answer. Cases must already match the
        plan's shape (see ``select_cases``) — this does not re-check it."""
        results: list[ReplayCaseResult] = []
        for case in cases:
            last_output: ToolOutput | None = None
            for tool_name, kwargs in case.actions:
                last_output = await self.tool_executor.execute(tool_name, context, **kwargs)
            replay_output = str(last_output.result) if last_output and last_output.success else ""
            agreement = agreement_score(replay_output, case.baseline_output)
            passed = agreement >= self.agreement_threshold
            results.append(ReplayCaseResult(
                user_input=case.user_input,
                agreement=agreement,
                passed=passed,
                replay_output=replay_output,
                baseline_output=case.baseline_output,
            ))

        total = len(results)
        passed_cases = sum(1 for r in results if r.passed)
        pass_rate = (passed_cases / total) if total else 0.0
        verdict = total > 0 and pass_rate >= self.pass_rate_threshold

        return ReplayReport(
            plan_id=plan.plan_id,
            intent=plan.intent,
            total_cases=total,
            passed_cases=passed_cases,
            pass_rate=pass_rate,
            agreement_threshold=self.agreement_threshold,
            pass_rate_threshold=self.pass_rate_threshold,
            verdict=verdict,
            cases=results,
        )


def select_cases(plan: CachedPlan, transcripts: list[ReactTranscript]) -> list[ReactTranscript]:
    """Historical transcripts whose executed tool-chain shape matches
    ``plan`` — the population a replay eval for this plan can draw on."""
    target_shape = plan_shape(plan.steps)
    return [t for t in transcripts if _transcript_shape(t) == target_shape]


def _transcript_shape(transcript: ReactTranscript) -> str:
    steps = [
        ToolChainStep(tool=name, kwargs_template={str(k): f"${k}" for k in kwargs})
        for name, kwargs in transcript.actions
    ]
    return plan_shape(steps)


async def save_report(workspace: Workspace, report: ReplayReport) -> Path:
    """Persist a replay-eval report to the workspace (D6's "报告落盘")."""
    relative = f"eval_reports/{report.plan_id}.json"
    return await workspace.write(relative, report.model_dump_json(indent=2))
