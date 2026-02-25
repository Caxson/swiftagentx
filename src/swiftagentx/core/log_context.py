"""
Request-level log context management.

Uses contextvars for thread-safe request ID propagation.
Automatically injects request_id into all log records via logging.Filter.

Usage:
    1. Call set_request_id("req_abc123") at request entry point
    2. All logger output will carry [request_id] prefix
    3. asyncio tasks inherit automatically (Python 3.7+)
    4. New threads need manual set_request_id() call
"""

import logging
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="")


def set_request_id(request_id: str) -> None:
    """Set request ID for current context (truncated to 12 chars for log readability)."""
    short_id = request_id.replace("-", "")[:12] if request_id else ""
    _request_id.set(short_id)


def get_request_id() -> str:
    """Get request ID for current context."""
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Log filter that injects request_id into every log record.

    Use %(request_id)s in your log format string.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get() or "-"  # type: ignore[attr-defined]
        return True
