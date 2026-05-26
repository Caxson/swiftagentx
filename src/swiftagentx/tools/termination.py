"""
Termination checker — determines when the ReAct loop should end.
"""

import logging
from collections.abc import Callable

from ..models.schema import SessionContext

logger = logging.getLogger(__name__)


class TerminationChecker:
    """
    Provides multiple termination condition checks for the ReAct loop.
    """

    def __init__(self) -> None:
        self.custom_checkers: list[Callable[[SessionContext], bool]] = []

    def check_max_iterations(self, context: SessionContext) -> bool:
        return context.current_iteration >= context.max_iterations

    def check_final_answer(self, last_thought: str) -> bool:
        final_markers = ["Final Answer:", "Answer:"]
        return any(marker in last_thought for marker in final_markers)

    def check_no_action(self, last_thought: str) -> bool:
        action_markers = ["Action:", "Tool:"]
        return not any(marker in last_thought for marker in action_markers)

    def check_repeated_actions(self, context: SessionContext, window: int = 3) -> bool:
        if len(context.steps) < window:
            return False
        recent_steps = context.steps[-window:]
        actions = [s.get("content", "") for s in recent_steps if s.get("type") == "ACTION"]
        return len(actions) >= 2 and len(actions) != len(set(actions))

    def check_error_state(self, context: SessionContext, error_threshold: int = 3) -> bool:
        error_count = sum(1 for step in context.steps if step.get("type") == "ERROR")
        return error_count >= error_threshold

    def add_custom_checker(self, checker: Callable[[SessionContext], bool]) -> None:
        self.custom_checkers.append(checker)

    def should_terminate(
        self,
        context: SessionContext,
        last_thought: str | None = None,
        check_max_iterations: bool = True,
        check_final_answer: bool = True,
        check_no_action: bool = False,
        check_repeated_actions: bool = False,
        check_error_state: bool = True,
    ) -> tuple[bool, str]:
        if check_max_iterations and self.check_max_iterations(context):
            return True, f"Reached max iterations: {context.max_iterations}"

        if check_final_answer and last_thought and self.check_final_answer(last_thought):
            return True, "Final answer marker detected"

        if check_no_action and last_thought and self.check_no_action(last_thought):
            return True, "No action found in thought"

        if check_repeated_actions and self.check_repeated_actions(context):
            return True, "Repeated actions detected"

        if check_error_state and self.check_error_state(context):
            return True, "Too many errors"

        for checker in self.custom_checkers:
            try:
                if checker(context):
                    return True, f"Custom checker triggered: {checker.__name__}"
            except Exception as exc:
                logger.warning(
                    "Custom termination checker %s raised %s: %s",
                    getattr(checker, "__name__", "<anonymous>"),
                    type(exc).__name__,
                    exc,
                )

        return False, "No termination condition met"

    def clear_custom_checkers(self) -> None:
        self.custom_checkers.clear()
