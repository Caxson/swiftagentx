"""
KnowledgeBaseStage — pipeline stage for knowledge base lookup.

Insert into the RequestPipeline to short-circuit on exact KB matches.
"""

import logging
from typing import Any, Dict

from ..core.pipeline import PipelineStage, StageAction, StageResult
from .base import KnowledgeBase

logger = logging.getLogger(__name__)


class KnowledgeBaseStage(PipelineStage):
    """
    Pipeline stage that queries the knowledge base.

    - If the top result score >= ``threshold``, returns SHORT_CIRCUIT
      with the document content as the answer.
    - Otherwise, stores search results in ``context["kb_results"]``
      and returns CONTINUE.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        threshold: float = 0.95,
        top_k: int = 5,
        name: str = "KnowledgeBaseStage",
    ):
        super().__init__(name=name)
        self.kb = kb
        self.threshold = threshold
        self.top_k = top_k

    async def execute(self, context: Dict[str, Any]) -> StageResult:
        user_input = context.get("user_input", "")
        if not user_input:
            return StageResult(action=StageAction.CONTINUE)

        try:
            results = await self.kb.search(user_input, top_k=self.top_k)
        except Exception as e:
            logger.warning(f"KB search failed, continuing pipeline: {e}")
            context["kb_results"] = []
            return StageResult(action=StageAction.CONTINUE)
        if not results:
            context["kb_results"] = []
            return StageResult(action=StageAction.CONTINUE)

        best = results[0]
        if best.score >= self.threshold:
            return StageResult(
                action=StageAction.SHORT_CIRCUIT,
                answer=best.document.content,
                metadata={"match_type": best.match_type, "score": best.score},
            )

        # Store results for downstream stages
        context["kb_results"] = [
            {"content": r.document.content, "score": r.score, "match_type": r.match_type}
            for r in results
        ]
        return StageResult(action=StageAction.CONTINUE)
