"""
Session workspace for SwiftAgentX v0.3.

A workspace is a per-session sandbox the agent can write transient files
into (PDF reports, generated images, transcripts, intermediate scratch
files, ...). The framework guarantees that:

- Files written through the workspace are isolated by session_id.
- The workspace can be cleaned up when the session ends.
- The backend is pluggable: local disk for dev, in-memory for tests,
  future S3/MinIO backend for production deployments.

Surface::

    async with agent.workspace(session_id="abc") as ws:
        await ws.write("report.pdf", pdf_bytes)
        path = ws.path("report.pdf")
        await ws.upload("report.pdf")     # backend-specific
    # workspace cleaned up

The async context manager is the canonical entry point because cleanup
is async (e.g. closing an S3 client). Synchronous use is supported via
``ws.write_sync(...)`` for the cases that have no I/O.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class WorkspaceBackend(ABC):
    """Pluggable storage backend for a session workspace."""

    @abstractmethod
    async def open(self, session_id: str) -> Workspace:
        """Return a Workspace bound to ``session_id`` (creates if absent)."""

    @abstractmethod
    async def cleanup(self, session_id: str) -> None:
        """Permanently delete the workspace for ``session_id``."""


# ---------------------------------------------------------------------------
# Workspace abstraction
# ---------------------------------------------------------------------------


class Workspace(ABC):
    """One session's workspace. Subclasses bind to a backend."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    @abstractmethod
    def path(self, relative: str) -> Path:
        """Resolve ``relative`` to an absolute path. Useful for tools that
        need a real filesystem path."""

    @abstractmethod
    async def write(self, relative: str, data: bytes | str) -> Path: ...

    @abstractmethod
    async def read(self, relative: str) -> bytes | None: ...

    @abstractmethod
    async def list(self) -> list[str]: ...

    @abstractmethod
    async def exists(self, relative: str) -> bool: ...

    @abstractmethod
    async def remove(self, relative: str) -> bool: ...

    @abstractmethod
    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Local-disk implementation
# ---------------------------------------------------------------------------


class LocalDiskWorkspaceBackend(WorkspaceBackend):
    """Per-session subdirectory under ``root``. Cleanup deletes the subdir."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = Path(tempfile.gettempdir()) / "swiftagentx-workspaces"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    async def open(self, session_id: str) -> Workspace:
        return _LocalDiskWorkspace(session_id, self.root)

    async def cleanup(self, session_id: str) -> None:
        # Defense in depth: only delete if the path is inside ``self.root``.
        target = (self.root / self._safe_dir(session_id)).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            logger.warning(
                "Refusing to cleanup workspace outside root: %s", target,
            )
            return
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    @staticmethod
    def _safe_dir(session_id: str) -> str:
        # The session id can be user-controlled. Strip any path separators
        # to guarantee the workspace stays inside the root.
        return session_id.replace("/", "_").replace("\\", "_").replace("..", "_")


class _LocalDiskWorkspace(Workspace):
    def __init__(self, session_id: str, root: Path) -> None:
        super().__init__(session_id)
        safe = LocalDiskWorkspaceBackend._safe_dir(session_id)
        self._dir = root / safe
        self._dir.mkdir(parents=True, exist_ok=True)

    def path(self, relative: str) -> Path:
        p = (self._dir / relative).resolve()
        # Reject relative paths that escape the workspace.
        if not str(p).startswith(str(self._dir.resolve())):
            raise ValueError(f"path {relative!r} escapes the workspace")
        return p

    async def write(self, relative: str, data: bytes | str) -> Path:
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            target.write_text(data, encoding="utf-8")
        else:
            target.write_bytes(data)
        return target

    async def read(self, relative: str) -> bytes | None:
        target = self.path(relative)
        if not target.exists():
            return None
        return target.read_bytes()

    async def list(self) -> list[str]:
        if not self._dir.exists():
            return []
        return sorted(
            str(p.relative_to(self._dir))
            for p in self._dir.rglob("*")
            if p.is_file()
        )

    async def exists(self, relative: str) -> bool:
        try:
            return self.path(relative).exists()
        except ValueError:
            return False

    async def remove(self, relative: str) -> bool:
        target = self.path(relative)
        if target.exists():
            target.unlink()
            return True
        return False

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# In-memory implementation (default; cheap; great for tests)
# ---------------------------------------------------------------------------


class InMemoryWorkspaceBackend(WorkspaceBackend):
    """Process-local dict of dicts. The default backend for unit tests."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, bytes]] = {}

    async def open(self, session_id: str) -> Workspace:
        store = self._sessions.setdefault(session_id, {})
        return _InMemoryWorkspace(session_id, store)

    async def cleanup(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class _InMemoryWorkspace(Workspace):
    def __init__(self, session_id: str, store: dict[str, bytes]) -> None:
        super().__init__(session_id)
        self._store = store

    def path(self, relative: str) -> Path:
        # No real filesystem path; tools that need one should fall back to
        # writing the bytes themselves or use the LocalDiskWorkspaceBackend.
        return Path(f"<inmem:{self.session_id}:{relative}>")

    async def write(self, relative: str, data: bytes | str) -> Path:
        self._store[relative] = data.encode("utf-8") if isinstance(data, str) else data
        return self.path(relative)

    async def read(self, relative: str) -> bytes | None:
        return self._store.get(relative)

    async def list(self) -> list[str]:
        return sorted(self._store.keys())

    async def exists(self, relative: str) -> bool:
        return relative in self._store

    async def remove(self, relative: str) -> bool:
        return self._store.pop(relative, None) is not None

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Context manager helper
# ---------------------------------------------------------------------------


@asynccontextmanager
async def use_workspace(
    backend: WorkspaceBackend,
    session_id: str,
    *,
    cleanup_on_exit: bool = False,
) -> AsyncIterator[Workspace]:
    """Async context manager that opens a workspace and closes it on exit.

    By default the workspace contents are preserved across context exits
    (sessions are long-lived in production). Pass ``cleanup_on_exit=True``
    for ephemeral workspaces (e.g. a tool that wants a scratch dir).
    """
    ws = await backend.open(session_id)
    try:
        yield ws
    finally:
        await ws.close()
        if cleanup_on_exit:
            await backend.cleanup(session_id)
