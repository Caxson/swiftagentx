"""
KnowledgeBaseTool — wraps a KnowledgeBase as a Tool for agent use.
"""

from typing import Any, Dict

from ..tools.base import AgentContext, Tool, ToolOutput, ToolOutputType
from .base import KnowledgeBase


class KnowledgeBaseTool(Tool):
    """
    Wraps a KnowledgeBase instance as an agent Tool.

    If a search result exceeds ``exact_match_threshold``, the result
    is returned as DIRECT_OUTPUT (short-circuiting LLM processing).
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        exact_match_threshold: float = 0.95,
        top_k: int = 5,
    ):
        super().__init__(
            name="knowledge_base",
            description="Search the knowledge base for relevant information",
            category="knowledge",
        )
        self.kb = kb
        self.exact_match_threshold = exact_match_threshold
        self.top_k = top_k

    async def execute(self, context: AgentContext, **kwargs: Any) -> ToolOutput:
        query = kwargs.get("query", "") or context.user_input
        if not query:
            return ToolOutput(success=False, result=None, error="No query provided")

        results = await self.kb.search(query, top_k=self.top_k)
        if not results:
            return ToolOutput(success=True, result="No relevant documents found.")

        # Exact/high-score match → direct output
        best = results[0]
        if best.score >= self.exact_match_threshold:
            return ToolOutput(
                success=True,
                result=best.document.content,
                output_type=ToolOutputType.DIRECT_OUTPUT,
                metadata={"match_type": best.match_type, "score": best.score},
            )

        # Format results for LLM processing
        formatted = "\n\n".join(
            f"[{r.match_type} score={r.score}] {r.document.content}"
            for r in results
        )
        return ToolOutput(
            success=True,
            result=formatted,
            output_type=ToolOutputType.LLM_PROCESSED,
            metadata={"result_count": len(results)},
        )

    def get_schema(self) -> Dict[str, Any]:
        schema = super().get_schema()
        schema["parameters"] = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        }
        return schema
