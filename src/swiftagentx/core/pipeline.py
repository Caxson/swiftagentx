"""
Request pipeline — configurable stage chain for request processing.

Default stages: Cache Check -> Intent Route -> Execute (ReAct/Scenario/Direct) -> Response
Users can insert custom stages (security check, KB lookup, etc.).
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StageAction(str, Enum):
    """What the pipeline should do after a stage completes."""
    CONTINUE = "continue"
    SHORT_CIRCUIT = "short_circuit"  # Skip remaining stages and return result
    ABORT = "abort"


class StageResult(BaseModel):
    """Result returned by a pipeline stage."""
    action: StageAction = StageAction.CONTINUE
    data: dict[str, Any] = {}
    answer: str | None = None  # If set and action is SHORT_CIRCUIT, this is the final answer
    metadata: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}


class PipelineStage(ABC):
    """
    Abstract pipeline stage.

    Implement `execute` to add custom processing logic.
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> StageResult:
        """
        Execute this pipeline stage.

        Args:
            context: Mutable context dict shared across all stages.
                Contains at minimum: user_input, session_id, user_id.
                Stages can read from and write to this dict.

        Returns:
            StageResult indicating whether to continue, short-circuit, or abort.
        """
        ...


class RequestPipeline:
    """
    Configurable request processing pipeline.

    Stages are executed in order. Each stage can:
    - CONTINUE: pass to next stage
    - SHORT_CIRCUIT: return immediately with a result (e.g., cache hit)
    - ABORT: stop with an error
    """

    def __init__(self) -> None:
        self.stages: list[PipelineStage] = []

    def add_stage(self, stage: PipelineStage) -> "RequestPipeline":
        self.stages.append(stage)
        return self

    def insert_stage(self, index: int, stage: PipelineStage) -> "RequestPipeline":
        self.stages.insert(index, stage)
        return self

    def remove_stage(self, name: str) -> bool:
        for i, stage in enumerate(self.stages):
            if stage.name == name:
                self.stages.pop(i)
                return True
        return False

    async def execute(self, context: dict[str, Any]) -> StageResult:
        """
        Execute all stages in sequence.

        Args:
            context: Shared mutable context dict.

        Returns:
            Final StageResult (from the last stage or a short-circuit).
        """
        last_result = StageResult()

        for stage in self.stages:
            try:
                logger.debug(f"Pipeline stage: {stage.name}")
                result = await stage.execute(context)

                # Merge stage data into context
                context.update(result.data)
                last_result = result

                if result.action == StageAction.SHORT_CIRCUIT:
                    logger.info(f"Pipeline short-circuited at stage: {stage.name}")
                    return result

                if result.action == StageAction.ABORT:
                    logger.warning(f"Pipeline aborted at stage: {stage.name}")
                    return result

            except Exception as e:
                logger.error(f"Pipeline stage '{stage.name}' failed: {e}", exc_info=True)
                return StageResult(
                    action=StageAction.ABORT,
                    data={"error": str(e), "failed_stage": stage.name},
                )

        return last_result

    def list_stages(self) -> list[str]:
        return [s.name for s in self.stages]
