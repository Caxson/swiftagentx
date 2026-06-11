"""
Intent router — classifies user intent and decides the execution path.

Three intent levels:
- REACT: Complex queries requiring multi-step reasoning and tool calls
- SCENARIO: High-frequency queries matching pre-defined tool chains
- DIRECT: Simple queries that can be answered directly by the LLM
"""

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .model_client import ModelClient, ModelResponse
from .retrieval import LexicalRetriever, ScenarioRetriever

logger = logging.getLogger(__name__)

# Above this many scenarios, the classifier prompt stops listing the full
# pool and shows only the retrieval top-K — accuracy of a light classifier
# degrades as the candidate list grows, retrieval doesn't.
DEFAULT_PREFILTER_TOP_K = 8


class IntentLevel(Enum):
    """Intent classification level."""
    REACT = 1       # Needs ReAct + multi-step tool calls
    SCENARIO = 2    # Matches a pre-defined scenario tool chain
    DIRECT = 3      # Direct LLM response (no tools needed)


class IntentResult(BaseModel):
    """Result of intent classification."""
    level: IntentLevel
    scenario: str | None = None
    slots: dict[str, str] = {}
    confidence: float = 0.0
    raw_output: str = ""
    metadata: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}


class IntentRouter:
    """
    Intent router — uses a light model to classify user intent.

    For dual-model strategy: use a fast/cheap model for classification (~200ms),
    then route to the appropriate execution path.
    """

    def __init__(
        self,
        scenarios: dict[str, Any] | None = None,
        retriever: ScenarioRetriever | None = None,
        prefilter_top_k: int = DEFAULT_PREFILTER_TOP_K,
    ):
        self._scenarios = scenarios or {}
        self._retriever: ScenarioRetriever = retriever if retriever is not None else LexicalRetriever()
        self._prefilter_top_k = max(1, prefilter_top_k)
        self._classification_prompt_template = (
            "Classify the user's intent into one of three levels:\n"
            "- level=1: Needs multi-step tool calls (e.g., complex queries, calculations)\n"
            "- level=2: Matches one of these scenarios: {scenario_list}\n"
            "- level=3: Can be answered directly without tools\n\n"
            "User input: {user_input}\n\n"
            "Respond with ONLY one line:\n"
            "  level=N scenario=<name> slots={{\"key\": \"value\"}}\n"
            "Rules:\n"
            "- Prefer level=2 over level=1 whenever a scenario covers the "
            "request, even if it involves multiple tools or steps.\n"
            "- Include scenario= and slots= ONLY when level=2.\n"
            "- A scenario's needed slots are shown after it as [slots: ...]. "
            "A slot value must be the SHORTEST literal span copied verbatim "
            "from the user input (a city name, an order id) — never a whole "
            "phrase. If the input doesn't explicitly state a slot's value, "
            "OMIT that slot entirely; never guess or fill placeholder text.\n"
            "- If level is 1 or 3, output just 'level=N'.\n"
            "Example: level=2 scenario=weather_query slots={{\"city\": \"北京\"}}\n"
            "Example (value absent): 今天天气怎么样 -> "
            "level=2 scenario=weather_query slots={{}}"
        )

    def set_classification_prompt(self, template: str) -> None:
        """Override the default classification prompt template."""
        self._classification_prompt_template = template

    def register_scenarios(self, scenarios: dict[str, Any]) -> None:
        """Register scenario names for classification."""
        self._scenarios.update(scenarios)

    async def classify(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
        model: ModelClient | None = None,
    ) -> IntentResult:
        """
        Classify user intent using the provided model.

        If no model is provided, defaults to DIRECT level.
        """
        if model is None:
            return IntentResult(level=IntentLevel.DIRECT, confidence=0.5, raw_output="no model provided")

        candidates = self._prefilter_scenarios(user_input)
        scenario_list = ", ".join(
            self._format_scenario_for_prompt(sid, info)
            for sid, info in candidates.items()
        )

        prompt = self._classification_prompt_template.format(
            scenario_list=scenario_list or "none",
            user_input=user_input,
        )

        try:
            response: ModelResponse = await model.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            result = self._parse_classification(response.content)
            if len(candidates) < len(self._scenarios):
                result.metadata["prefilter"] = {
                    "pool": len(self._scenarios),
                    "shown": list(candidates),
                }
            return result
        except Exception as e:
            logger.error(f"Intent classification failed: {e}", exc_info=True)
            return IntentResult(level=IntentLevel.DIRECT, confidence=0.3, raw_output=str(e))

    def _prefilter_scenarios(self, user_input: str) -> dict[str, Any]:
        """Keep the classification prompt O(K) regardless of pool size.

        With ≤K scenarios this is a no-op — small deployments see the exact
        same prompt as before. A scenario wrongly filtered out can't be
        picked by the classifier, so the request degrades to ReAct (slower
        but still correct), never to a scenario misfire. A failing custom
        retriever falls back to the full pool, again preserving the old
        behavior instead of crashing classification.
        """
        if len(self._scenarios) <= self._prefilter_top_k:
            return self._scenarios

        docs = {
            sid: self._scenario_document(sid, info)
            for sid, info in self._scenarios.items()
        }
        try:
            ranked = self._retriever.rank(user_input, docs)
        except Exception:
            logger.error(
                "Scenario prefilter failed; falling back to full pool",
                exc_info=True,
            )
            return self._scenarios

        keep = [sid for sid in ranked if sid in self._scenarios][: self._prefilter_top_k]
        logger.debug(
            f"Scenario prefilter: {len(self._scenarios)} -> {len(keep)} candidates: {keep}"
        )
        return {sid: self._scenarios[sid] for sid in keep}

    @staticmethod
    def _scenario_document(sid: str, info: Any) -> str:
        """Build the retrieval anchor text for one scenario.

        Triggers are the strongest anchors — they are real phrasings the
        scenario author expects users to type — followed by name,
        description and slot names.
        """
        if not isinstance(info, dict):
            return f"{sid} {info}"
        parts: list[str] = [sid, str(info.get("name", "")), str(info.get("description", ""))]
        parts.extend(str(t) for t in info.get("triggers") or [])
        parts.extend(str(s) for s in info.get("slots") or [])
        return " ".join(p for p in parts if p)

    @staticmethod
    def _format_scenario_for_prompt(sid: str, info: Any) -> str:
        """Render one scenario for the classifier prompt: id, name, a short
        description and its slots.

        The description is load-bearing: with only ``id(name)[slots]`` the
        classifier routinely judged multi-tool requests as level=1 even
        when a scenario covered them exactly. Capped so a top-K candidate
        list stays bounded.
        """
        if isinstance(info, dict):
            name = info.get("name", sid)
            desc = str(info.get("description") or "").strip()
            desc_str = f": {desc[:60]}" if desc else ""
            slots = info.get("slots") or []
            slot_str = f" [slots: {', '.join(slots)}]" if slots else ""
            return f"{sid}({name}{desc_str}){slot_str}"
        return f"{sid}({info})"

    def _parse_classification(self, raw_output: str) -> IntentResult:
        """Parse LLM output into IntentResult."""
        raw = raw_output.strip().lower()

        level = IntentLevel.DIRECT
        scenario = None
        slots: dict[str, str] = {}
        confidence = 0.5

        if "level=1" in raw:
            level = IntentLevel.REACT
            confidence = 0.8
        elif "level=2" in raw:
            level = IntentLevel.SCENARIO
            confidence = 0.8
            # Extract scenario name
            import re
            match = re.search(r"scenario=(\w+)", raw)
            if match:
                scenario = match.group(1)
                if scenario in self._scenarios:
                    confidence = 0.9
                    # Slots are parsed from the ORIGINAL (case-preserving)
                    # output — lowercasing would mangle values like city
                    # names or IDs.
                    slots = self._parse_slots(raw_output)
                else:
                    # Scenario not found, fall back to REACT
                    level = IntentLevel.REACT
                    scenario = None
                    confidence = 0.6
        elif "level=3" in raw:
            level = IntentLevel.DIRECT
            confidence = 0.8

        return IntentResult(
            level=level,
            scenario=scenario,
            slots=slots,
            confidence=confidence,
            raw_output=raw_output,
        )

    @staticmethod
    def _parse_slots(raw_output: str) -> dict[str, str]:
        """Pull ``slots={...}`` out of the classifier output, tolerantly.

        Small/cheap classification models don't always emit clean JSON, so
        we try ``json.loads`` on the ``{...}`` blob first and fall back to a
        forgiving ``key:value`` / ``key=value`` pair scan.
        """
        import json
        import re

        m = re.search(r"slots\s*=\s*(\{.*?\})", raw_output, re.DOTALL)
        if not m:
            return {}
        blob = m.group(1)
        try:
            data = json.loads(blob)
            if isinstance(data, dict):
                return {
                    str(k): str(v).strip()
                    for k, v in data.items()
                    if v not in (None, "")
                }
        except (json.JSONDecodeError, ValueError):
            pass
        out: dict[str, str] = {}
        for key, value in re.findall(r'"?(\w+)"?\s*[:=]\s*"?([^",}]+)"?', blob):
            value = value.strip()
            if value:
                out[key] = value
        return out
