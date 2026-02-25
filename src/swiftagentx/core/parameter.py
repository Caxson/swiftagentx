"""
Parameter management — global and session-level parameters with thread safety.
"""

from typing import Any, Dict, Optional, Type
from threading import RLock


class ParameterManager:
    """
    Parameter manager — supports global and session-scoped parameters.
    """

    def __init__(self) -> None:
        self.global_params: Dict[str, Dict[str, Any]] = {}
        self.session_params: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def register_global_param(
        self, name: str, type_: Type = str, default: Any = None, description: str = ""
    ) -> None:
        with self._lock:
            if name not in self.global_params:
                self.global_params[name] = {"type": type_, "description": description, "value": default}

    def set_global_param(self, name: str, value: Any) -> None:
        with self._lock:
            if name not in self.global_params:
                self.register_global_param(name)
            self.global_params[name]["value"] = value

    def get_global_param(self, name: str, default: Any = None) -> Any:
        with self._lock:
            if name in self.global_params:
                return self.global_params[name].get("value", default)
            return default

    def set_session_param(self, session_id: str, name: str, value: Any) -> None:
        with self._lock:
            if session_id not in self.session_params:
                self.session_params[session_id] = {}
            self.session_params[session_id][name] = value

    def get_session_param(self, session_id: str, name: str, default: Any = None) -> Any:
        with self._lock:
            if session_id in self.session_params:
                return self.session_params[session_id].get(name, default)
            return default

    def get_all_session_params(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return self.session_params.get(session_id, {}).copy()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self.session_params.pop(session_id, None)

    def get_merged_params(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            merged = {name: cfg.get("value") for name, cfg in self.global_params.items()}
            if session_id in self.session_params:
                merged.update(self.session_params[session_id])
            return merged

    def clear_all(self) -> None:
        with self._lock:
            self.global_params.clear()
            self.session_params.clear()


_global_parameter_manager: Optional[ParameterManager] = None


def get_parameter_manager() -> ParameterManager:
    global _global_parameter_manager
    if _global_parameter_manager is None:
        _global_parameter_manager = ParameterManager()
    return _global_parameter_manager


def init_parameter_manager() -> ParameterManager:
    global _global_parameter_manager
    _global_parameter_manager = ParameterManager()
    return _global_parameter_manager
