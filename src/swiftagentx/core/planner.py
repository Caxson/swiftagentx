"""
Planner fast path — one light-model call turns a REACT-level request into a
deterministic, templated tool plan.

Where ReAct spends N+1 LLM calls deciding each step as it goes, the Planner
spends ONE light call emitting the whole chain up front, then executes it
deterministically through the same engine that runs Scenarios. A plan that
keeps succeeding graduates into a real Scenario:

    REACT request
      -> PlanStore.match()  (cached plan reuse: 0 planning calls)
      -> Planner.generate() (fresh plan: 1 light call)
      -> validate -> deterministic execution
      -> success: probation cache; after N successes auto-promote
         (rule track) or wait for promote_plan() (manual track)
      -> promoted: registered as a Scenario — the classifier handles it
         like any human-authored scenario from then on.

Any failure at any stage falls back to the normal ReAct loop — the fast
path can only make requests faster, never wronger.

Plans are TEMPLATED (steps reference ``$slot`` vars, with this request's
values carried separately) so one plan is reusable across phrasings, and a
promoted plan is already a valid ``ScenarioConfig``.
"""

import hashlib
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from ..tools.scenario import ScenarioConfig, ToolChainStep
from .retrieval import tokenize

logger = logging.getLogger(__name__)

# Vars the agent always supplies at execution time — plans may reference
# them without declaring a slot (mirrors ScenarioConfig.required_vars).
RESERVED_VARS = {"user_input", "user_id", "session_id"}

_INTENT_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
_VAR_RE = re.compile(r"\$(\w+)")


class GeneratedPlan(BaseModel):
    """A templated tool plan plus this request's slot values."""

    intent: str
    description: str = ""
    steps: list[ToolChainStep] = Field(default_factory=list)
    slots: dict[str, str] = Field(default_factory=dict)

    def required_vars(self) -> set[str]:
        """Template vars the plan needs from outside (same rules as Scenario)."""
        produced = {s.extract_to for s in self.steps if s.extract_to}
        names: set[str] = set()
        for step in self.steps:
            templates = list(step.kwargs_template.values())
            if step.query_template:
                templates.append(step.query_template)
            for tmpl in templates:
                names.update(_VAR_RE.findall(tmpl))
        return names - RESERVED_VARS - produced


class Planner:
    """Builds the planning prompt, parses and validates the LLM's plan."""

    PROMPT_TEMPLATE = (
        "You are a tool-call planner. Decide whether the user's request can "
        "be served by a FIXED, LINEAR sequence of the tools below (no "
        "branching, no decisions that depend on a tool's output).\n\n"
        "Available tools:\n{tool_list}\n\n"
        "User request: {user_input}\n\n"
        "If plannable, respond with ONLY a JSON object:\n"
        '{{"intent": "<short_snake_case_name>", "description": "<one line>", '
        '"steps": [{{"tool": "<name>", "kwargs": {{"param": "$slot_or_$prev"}}, '
        '"extract_to": "<var_name_optional>"}}], '
        '"slots": {{"slot": "<value copied verbatim from the request>"}}}}\n'
        "Rules:\n"
        "- Use ONLY the listed tools, at most {max_steps} steps.\n"
        "- Parameterize user-specific values as $slots; a later step may "
        "reference an earlier step's extract_to var as $var.\n"
        "- Slot values must be the shortest literal span copied verbatim "
        "from the request. If a needed value is absent, the request is NOT "
        "plannable.\n"
        '- If not confidently plannable, respond with exactly: {{"intent": null}}'
    )

    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps

    def build_prompt(self, user_input: str, tool_schemas: dict[str, dict]) -> str:
        lines = []
        for name, schema in tool_schemas.items():
            params = ", ".join((schema.get("parameters") or {}).get("properties") or {})
            desc = schema.get("description", "")
            lines.append(f"- {name}({params}): {desc}")
        return self.PROMPT_TEMPLATE.format(
            tool_list="\n".join(lines) or "(none)",
            user_input=user_input,
            max_steps=self.max_steps,
        )

    def parse(self, raw_output: str) -> GeneratedPlan | None:
        """Tolerantly extract a GeneratedPlan from LLM output.

        Returns None for "not plannable" replies and for anything that
        doesn't parse — the caller falls back to ReAct either way.
        """
        data = _extract_json_object(raw_output)
        if not isinstance(data, dict) or not data.get("intent"):
            return None
        try:
            steps = [
                ToolChainStep(
                    tool=str(s["tool"]),
                    kwargs_template={
                        str(k): str(v)
                        for k, v in (s.get("kwargs") or s.get("kwargs_template") or {}).items()
                    },
                    extract_to=str(s.get("extract_to") or ""),
                )
                for s in data.get("steps") or []
            ]
            slots = {
                str(k): str(v).strip()
                for k, v in (data.get("slots") or {}).items()
                if str(v).strip()
            }
            return GeneratedPlan(
                intent=_sanitize_intent(str(data["intent"])),
                description=str(data.get("description") or ""),
                steps=steps,
                slots=slots,
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Planner output unparsable: {e}")
            return None

    def validate(self, plan: GeneratedPlan, available_tools: set[str]) -> str | None:
        """Return an error string, or None if the plan is executable."""
        if not plan.steps:
            return "plan has no steps"
        if len(plan.steps) > self.max_steps:
            return f"plan has {len(plan.steps)} steps (max {self.max_steps})"

        produced: set[str] = set()
        for i, step in enumerate(plan.steps):
            if step.tool not in available_tools:
                return f"step {i + 1} uses unknown tool '{step.tool}'"
            templates = list(step.kwargs_template.values())
            if step.query_template:
                templates.append(step.query_template)
            for tmpl in templates:
                for var in _VAR_RE.findall(tmpl):
                    if (var not in RESERVED_VARS and var not in plan.slots
                            and var not in produced):
                        return f"step {i + 1} references undefined var '${var}'"
            if step.extract_to:
                produced.add(step.extract_to)
        return None


class CachedPlan(BaseModel):
    """A plan in the reuse cache, with its promotion bookkeeping."""

    plan_id: str
    intent: str
    description: str = ""
    steps: list[ToolChainStep] = Field(default_factory=list)
    slot_names: list[str] = Field(default_factory=list)
    # Real user phrasings that produced/reused this plan — these become the
    # Scenario triggers after promotion.
    source_queries: list[str] = Field(default_factory=list)
    # The same phrasings with their slot VALUES stripped out. Matching runs
    # against these: "帮我查北京天气" and "帮我查上海天气" share the same
    # template but their raw bigrams barely overlap (the value perturbs
    # every bigram around it), so anchors must be value-free.
    match_anchors: list[str] = Field(default_factory=list)
    successes: int = 0
    failures: int = 0
    # Reuse gate: only approved plans are matched against new requests.
    # auto_reuse stores approve on add; otherwise a developer approves.
    approved: bool = False
    promoted: bool = False


class PlanStore:
    """Candidate cache for generated plans, with two promotion gates.

    A successful plan's lifecycle has two independently gated steps, each
    with a rule track and a manual track:

    1. REUSE — may this plan be matched against future requests at all?
       ``auto_reuse=True``: approved on entry (rule track).
       ``auto_reuse=False`` (manual): the plan only accumulates stats as a
       candidate (same-shape regenerations dedupe into it) until the
       developer approves it. Unapproved plans are one-shot accelerators.
    2. PROMOTION to Scenario — after ``promote_after`` clean successes the
       plan is reported promotable (the agent auto-registers it when
       ``plan_auto_promote`` is on); otherwise ``Agent.promote_plan()`` /
       ``export_plan_scenario()`` keep it a developer decision.

    Matching uses anchor containment over the same CJK-aware tokens the
    scenario prefilter uses: anchors are stored with their slot values
    stripped (the value perturbs every bigram around it), and a query
    matches when it covers >= ``match_threshold`` of an anchor's tokens.
    The bar is deliberately conservative — a false match would run the
    WRONG tools — and is further guarded downstream: every required slot
    must extract from the new phrasing and every step must succeed, or
    the request falls back to ReAct. Misses just cost one planning call.
    """

    def __init__(
        self,
        max_size: int = 256,
        match_threshold: float = 0.6,
        promote_after: int = 3,
        evict_after_failures: int = 2,
        auto_reuse: bool = True,
    ):
        self._plans: dict[str, CachedPlan] = {}
        self.max_size = max(1, max_size)
        self.match_threshold = match_threshold
        self.promote_after = max(1, promote_after)
        self.evict_after_failures = max(1, evict_after_failures)
        self.auto_reuse = auto_reuse

    def add(self, plan: GeneratedPlan, source_query: str) -> CachedPlan:
        """Cache a freshly generated plan (deduped by tool-chain shape)."""
        existing = self._find_same_shape(plan)
        if existing is not None:
            self._add_anchor(existing, source_query, plan.slots)
            return existing

        plan_id = self._make_id(plan)
        cached = CachedPlan(
            plan_id=plan_id,
            intent=plan.intent,
            description=plan.description,
            steps=list(plan.steps),
            slot_names=sorted(plan.required_vars()),
            approved=self.auto_reuse,
        )
        self._add_anchor(cached, source_query, plan.slots)
        self._evict_if_full()
        self._plans[plan_id] = cached
        logger.info(f"Plan cached: {plan_id} ({len(plan.steps)} steps)")
        return cached

    def match(self, query: str) -> CachedPlan | None:
        """Best approved, unpromoted plan whose anchor the query covers."""
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return None
        best: CachedPlan | None = None
        best_score = 0.0
        for plan in self._plans.values():
            if not plan.approved:
                continue  # reuse gate closed: candidate awaiting review
            if plan.promoted:
                continue  # promoted plans are served by the Scenario path
            for anchor in plan.match_anchors:
                anchor_tokens = set(tokenize(anchor))
                # A near-empty anchor (the whole query was a slot value)
                # would make containment trivially high — skip it.
                if len(anchor_tokens) < 2:
                    continue
                score = len(query_tokens & anchor_tokens) / len(anchor_tokens)
                if score > best_score:
                    best, best_score = plan, score
        if best is not None and best_score >= self.match_threshold:
            return best
        return None

    def record_success(
        self,
        plan_id: str,
        source_query: str = "",
        slots: dict[str, str] | None = None,
    ) -> bool:
        """Record a successful execution. Returns True when the plan just
        became promotable under the rule track."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return False
        plan.successes += 1
        if source_query:
            self._add_anchor(plan, source_query, slots or {})
        return (not plan.promoted
                and plan.failures == 0
                and plan.successes >= self.promote_after)

    def record_failure(self, plan_id: str) -> None:
        """Record a failed execution; evict probation plans that keep failing."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return
        plan.failures += 1
        if not plan.promoted and plan.failures >= self.evict_after_failures:
            del self._plans[plan_id]
            logger.info(f"Plan evicted after {plan.failures} failures: {plan_id}")

    def approve(self, plan_id: str) -> bool:
        """Open the reuse gate for a candidate plan (manual track)."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return False
        plan.approved = True
        return True

    def mark_promoted(self, plan_id: str) -> None:
        plan = self._plans.get(plan_id)
        if plan is not None:
            plan.approved = True  # promotion implies the reuse gate
            plan.promoted = True

    def get(self, plan_id: str) -> CachedPlan | None:
        return self._plans.get(plan_id)

    def list_plans(self) -> list[CachedPlan]:
        return list(self._plans.values())

    def to_scenario_config(self, plan_id: str) -> ScenarioConfig | None:
        """Render a cached plan as a ScenarioConfig — the promotion artifact.

        The plan's real user phrasings become the scenario's triggers, so
        the retrieval prefilter gets anchors that actual users typed —
        usually better anchors than hand-written descriptions.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            return None
        return ScenarioConfig(
            name=plan.intent,
            description=plan.description or f"Auto-promoted plan {plan.plan_id}",
            triggers=plan.source_queries[:8],
            tool_chain=list(plan.steps),
        )

    def _find_same_shape(self, plan: GeneratedPlan) -> CachedPlan | None:
        shape = _plan_shape(plan.steps)
        for cached in self._plans.values():
            if _plan_shape(cached.steps) == shape:
                return cached
        return None

    @staticmethod
    def _add_anchor(
        plan: CachedPlan, query: str, slots: dict[str, str], cap: int = 8
    ) -> None:
        if not query or query in plan.source_queries or len(plan.source_queries) >= cap:
            return
        plan.source_queries.append(query)
        plan.match_anchors.append(_strip_slot_values(query, slots))

    def _evict_if_full(self) -> None:
        if len(self._plans) < self.max_size:
            return
        for pid, plan in list(self._plans.items()):
            if not plan.promoted:
                del self._plans[pid]
                logger.info(f"Plan cache full ({self.max_size}); evicted {pid}")
                return

    @staticmethod
    def _make_id(plan: GeneratedPlan) -> str:
        digest = hashlib.sha1(_plan_shape(plan.steps).encode()).hexdigest()[:8]
        return f"plan_{plan.intent}_{digest}"


def _strip_slot_values(text: str, slots: dict[str, str]) -> str:
    """Remove slot VALUES from a phrasing, leaving its reusable template.

    Longest values first so overlapping values ("北京市" before "北京")
    strip cleanly. The replacement space breaks CJK bigram runs at the
    removal point, so no spurious cross-boundary bigrams survive.
    """
    for value in sorted(slots.values(), key=len, reverse=True):
        if value:
            text = text.replace(value, " ")
    return text


def _plan_shape(steps: list[ToolChainStep]) -> str:
    """Canonical signature of a tool chain: tools + param names, not values."""
    return "|".join(
        f"{s.tool}({','.join(sorted(s.kwargs_template))})>{s.extract_to}"
        for s in steps
    )


def _sanitize_intent(intent: str) -> str:
    candidate = re.sub(r"[^a-z0-9_]", "_", intent.strip().lower()).strip("_")[:40]
    return candidate if _INTENT_RE.match(candidate) else "plan"


def _extract_json_object(raw: str) -> Any:
    """Pull the first JSON object out of possibly-noisy LLM output."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
    return None
