"""
Tool registry — manages all available tools.
"""

from typing import Dict, List, Optional
from threading import RLock
from .base import Tool


class ToolRegistry:
    """
    Tool registry for registering, querying, and managing tools.
    """

    def __init__(self) -> None:
        self.tools: Dict[str, Tool] = {}
        self.tool_categories: Dict[str, List[str]] = {}
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

    def get(self, tool_name: str) -> Optional[Tool]:
        with self._lock:
            return self.tools.get(tool_name)

    def get_all(self) -> Dict[str, Tool]:
        with self._lock:
            return self.tools.copy()

    def get_by_category(self, category: str) -> List[Tool]:
        with self._lock:
            tool_names = self.tool_categories.get(category, [])
            return [self.tools[name] for name in tool_names if name in self.tools]

    def list_tools(self) -> List[str]:
        with self._lock:
            return list(self.tools.keys())

    def list_categories(self) -> List[str]:
        with self._lock:
            return list(self.tool_categories.keys())

    def get_schemas(self) -> Dict[str, dict]:
        with self._lock:
            return {name: tool.get_schema() for name, tool in self.tools.items()}

    def clear(self) -> None:
        with self._lock:
            self.tools.clear()
            self.tool_categories.clear()

    def count(self) -> int:
        with self._lock:
            return len(self.tools)
