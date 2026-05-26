"""
Tool registry — manages all available tools.
"""

from threading import RLock

from .base import Tool


class ToolRegistry:
    """
    Tool registry for registering, querying, and managing tools.
    """

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}
        self.tool_categories: dict[str, list[str]] = {}
        self._lock = RLock()

    def register(self, tool: Tool) -> None:
        with self._lock:
            if tool.name in self.tools:
                raise ValueError(f"Tool '{tool.name}' is already registered")
            self.tools[tool.name] = tool
            if tool.category not in self.tool_categories:
                self.tool_categories[tool.category] = []
            self.tool_categories[tool.category].append(tool.name)

    def unregister(self, tool_name: str) -> bool:
        with self._lock:
            if tool_name in self.tools:
                tool = self.tools.pop(tool_name)
                if tool.category in self.tool_categories:
                    self.tool_categories[tool.category].remove(tool_name)
                return True
            return False

    def get(self, tool_name: str) -> Tool | None:
        with self._lock:
            return self.tools.get(tool_name)

    def get_all(self) -> dict[str, Tool]:
        with self._lock:
            return self.tools.copy()

    def get_by_category(self, category: str) -> list[Tool]:
        with self._lock:
            tool_names = self.tool_categories.get(category, [])
            return [self.tools[name] for name in tool_names if name in self.tools]

    def list_tools(self) -> list[str]:
        with self._lock:
            return list(self.tools.keys())

    def list_categories(self) -> list[str]:
        with self._lock:
            return list(self.tool_categories.keys())

    def get_schemas(self) -> dict[str, dict]:
        with self._lock:
            return {name: tool.get_schema() for name, tool in self.tools.items()}

    def clear(self) -> None:
        with self._lock:
            self.tools.clear()
            self.tool_categories.clear()

    def count(self) -> int:
        with self._lock:
            return len(self.tools)

    # ------------------------------------------------------------------
    # Lazy tool loading (v0.3)
    # ------------------------------------------------------------------

    def select_tools_for_query(
        self,
        query: str,
        *,
        threshold: int = 20,
        max_returned: int = 8,
    ) -> list[Tool]:
        """
        Return a subset of tools relevant to ``query``.

        When the registry has fewer than ``threshold`` tools, returns all
        of them (the prompt is small enough already). Above the threshold,
        scores each tool against ``query`` by simple keyword overlap
        between the query and the tool's name + description + category;
        returns the top ``max_returned``.

        This is the framework's lightweight lazy-tool-loading: a LIGHT
        classifier is not consulted here — the score is computed locally,
        deterministically, and fast. A future variant can plug in an
        embedding model.
        """
        with self._lock:
            tools = list(self.tools.values())

        if len(tools) <= threshold:
            return tools

        terms = {t.lower() for t in query.split() if len(t) > 2}
        if not terms:
            return tools[:max_returned]

        def score(tool: Tool) -> int:
            haystack = f"{tool.name} {tool.description} {tool.category}".lower()
            return sum(1 for term in terms if term in haystack)

        ranked = sorted(tools, key=score, reverse=True)
        # Drop tools with score 0 (none of the query terms match).
        scored = [t for t in ranked if score(t) > 0]
        if not scored:
            return ranked[:max_returned]
        return scored[:max_returned]

    def schemas_for_query(
        self,
        query: str,
        *,
        threshold: int = 20,
        max_returned: int = 8,
    ) -> dict[str, dict]:
        """Like :meth:`get_schemas` but filtered via :meth:`select_tools_for_query`."""
        tools = self.select_tools_for_query(
            query, threshold=threshold, max_returned=max_returned,
        )
        return {t.name: t.get_schema() for t in tools}
