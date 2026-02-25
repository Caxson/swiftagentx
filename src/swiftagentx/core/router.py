"""
Intent router — classifies user intent and decides the execution path.

Three intent levels:
- REACT: Complex queries requiring multi-step reasoning and tool calls
- SCENARIO: High-frequency queries matching pre-defined tool chains
- DIRECT: Simple queries that can be answered directly by the LLM
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import logging

from .model_client import ModelClient, ModelResponse

logger = logging.getLogger(__name__)


class IntentLevel(Enum):
    """Intent classification level."""
    REACT = 1       # Needs ReAct + multi-step tool calls
    SCENARIO = 2    # Matches a pre-defined scenario tool chain
    DIRECT = 3      # Direct LLM response (no tools needed)


class IntentResult(BaseModel):
    """Result of intent classification."""
    level: IntentLevel
    scenario: Optional[str] = None
    confidence: float = 0.0
    raw_output: str = ""
    metadata: Dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}


class IntentRouter:
    """
    Intent router — uses a light model to classify user intent.

    For dual-model strategy: use a fast/cheap model for classification (~200ms),
    then route to the appropriate execution path.
    """

    def __init__(self, scenarios: Optional[Dict[str, Any]] = None):
        self._scenarios = scenarios or {}
        self._classification_prompt_template = (
            "Classify the user's intent into one of three levels:\n"
            "- level=1: Needs multi-step tool calls (e.g., complex queries, calculations)\n"
            "- level=2: Matches a scenario: {scenario_list}\n"
            "- level=3: Can be answered directly without tools\n\n"
            "User input: {user_input}\n\n"
            "Respond with ONLY: level=N scenario=name\n"
            "If level is not 2, omit the scenario part."
        )

    def set_classification_prompt(self, template: str) -> None:
        """Override the default classification prompt template."""
        self._classification_prompt_template = template

    def register_scenarios(self, scenarios: Dict[str, Any]) -> None:
        """Register scenario names for classification."""
        self._scenarios.update(scenarios)

    async def classify(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        model: Optional[ModelClient] = None,
    ) -> IntentResult:
        """
        Classify user intent using the provided model.

        If no model is provided, defaults to DIRECT level.
        """
        if model is None:
            return IntentResult(level=IntentLevel.DIRECT, confidence=0.5, raw_output="no model provided")

        scenario_list = ", ".join(
            f"{sid}({info.get('name', sid) if isinstance(info, dict) else sid})"
            for sid, info in self._scenarios.items()
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
            return self._parse_classification(response.content)
        except Exception as e:
            logger.error(f"Intent classification failed: {e}", exc_info=True)
            return IntentResult(level=IntentLevel.DIRECT, confidence=0.3, raw_output=str(e))

    def _parse_classification(self, raw_output: str) -> IntentResult:
        """Parse LLM output into IntentResult."""
        raw = raw_output.strip().lower()

        level = IntentLevel.DIRECT
        scenario = None
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
            confidence=confidence,
            raw_output=raw_output,
        )
