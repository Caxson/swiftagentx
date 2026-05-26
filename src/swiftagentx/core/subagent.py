"""
Sub-agent dispatch for SwiftAgentX v0.3.

A sub-agent is a focused agent with a bounded mission, its own tool
subset, and an isolated context. The main agent can dispatch one or many
in parallel; only the structured results come back. The main agent's
context stays clean of the sub-agents' intermediate tool calls.

Typical use: a customer-service ReAct iteration needs three independent
lookups (account history, recent orders, open tickets). Instead of
serializing them as three rounds of ReAct (which pollutes the main
prompt with intermediate observations), the agent dispatches three
sub-agents in parallel and merges only the structured results.

Roles are declarative: register a ``SubAgentRole`` with a system prompt,
allowed tool list, model tier, and result schema; thereafter the role
can be dispatched by name.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..models.config import ModelTier
from .model_client import ModelClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubAgentRole:
    """Declarative description of a sub-agent role.

    A role binds a name to (system_prompt, allowed_tools, model_tier,
    timeout). The main agent looks up the role by name when
    ``dispatch_subagents`` is called.
    """

    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    model_tier: ModelTier = ModelTier.LIGHT
    timeout_seconds: float = 20.0
    max_iterations: int = 3


class SubAgentRequest(BaseModel):
    """One sub-agent dispatch — pairs a role with its specific input."""

    role: str
    input: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubAgentResult(BaseModel):
    """The structured return value from a sub-agent invocation."""

    role: str
    success: bool
    output: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Role registry + handler protocol
# ---------------------------------------------------------------------------


SubAgentHandler = Callable[[SubAgentRequest, "SubAgentInvocation"], Awaitable[SubAgentResult]]
"""A handler executes one role's mission.

The framework ships a default handler that uses the agent's HEAVY/LIGHT
model directly with the role's allowed tool subset; advanced users can
register custom handlers for roles that need more control (e.g. a sub-
agent that itself spawns sub-sub-agents, or one that queries an
external service rather than an LLM).
"""


@dataclass
class SubAgentInvocation:
    """Context passed to a sub-agent handler."""

    role: SubAgentRole
    parent_agent: Any  # the Agent instance
    request: SubAgentRequest
    request_id: str
    started_at: float


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SubAgentManager:
    """
    Holds role definitions and dispatches sub-agent requests.

    Lifecycle:

      manager.register_role(role)
      results = await manager.dispatch(parent_agent, [req1, req2, ...])

    Roles can also be registered via ``register_handler`` to plug in a
    custom (non-LLM) execution path — useful when the "sub-agent" is
    actually a deterministic data lookup.
    """

    def __init__(self) -> None:
        self._roles: dict[str, SubAgentRole] = {}
        self._handlers: dict[str, SubAgentHandler] = {}

    # ---- registration ----------------------------------------------------

    def register_role(
        self, role: SubAgentRole, handler: SubAgentHandler | None = None,
    ) -> None:
        if role.name in self._roles:
            logger.info("Overwriting existing sub-agent role %s", role.name)
        self._roles[role.name] = role
        if handler is not None:
            self._handlers[role.name] = handler

    def unregister_role(self, name: str) -> bool:
        existed = name in self._roles
        self._roles.pop(name, None)
        self._handlers.pop(name, None)
        return existed

    def get_role(self, name: str) -> SubAgentRole | None:
        return self._roles.get(name)

    def list_roles(self) -> list[str]:
        return list(self._roles.keys())

    # ---- dispatch --------------------------------------------------------

    async def dispatch(
        self,
        parent_agent: Any,
        requests: list[SubAgentRequest],
        *,
        timeout_seconds: float | None = None,
    ) -> list[SubAgentResult]:
        """
        Dispatch ``requests`` in parallel and return one result per request.

        Each sub-agent runs concurrently via ``asyncio.gather``. A failed
        sub-agent (exception, timeout, missing role) produces a
        ``SubAgentResult`` with ``success=False`` — it never propagates
        an exception to the caller, so a single bad sub-agent doesn't
        break the fan-out.
        """
        tasks = [self._dispatch_one(parent_agent, req, timeout_seconds) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _dispatch_one(
        self,
        parent_agent: Any,
        request: SubAgentRequest,
        timeout_seconds: float | None,
    ) -> SubAgentResult:
        role = self._roles.get(request.role)
        if role is None:
            return SubAgentResult(
                role=request.role, success=False,
                error=f"unknown sub-agent role: {request.role!r}",
            )
        handler = self._handlers.get(request.role, default_subagent_handler)
        started = time.perf_counter()
        invocation = SubAgentInvocation(
            role=role,
            parent_agent=parent_agent,
            request=request,
            request_id=str(uuid.uuid4()),
            started_at=started,
        )
        effective_timeout = timeout_seconds or role.timeout_seconds
        try:
            result = await asyncio.wait_for(handler(request, invocation),
                                            timeout=effective_timeout)
        except (TimeoutError, asyncio.TimeoutError):
            return SubAgentResult(
                role=request.role, success=False,
                error=f"timeout after {effective_timeout:.1f}s",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Sub-agent %s handler raised", request.role)
            return SubAgentResult(
                role=request.role, success=False, error=str(exc),
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        # Some handlers may forget to populate duration_ms.
        if result.duration_ms == 0.0:
            result.duration_ms = (time.perf_counter() - started) * 1000
        return result


# ---------------------------------------------------------------------------
# Default handler — direct LLM call with the role's tool subset
# ---------------------------------------------------------------------------


async def default_subagent_handler(
    request: SubAgentRequest,
    invocation: SubAgentInvocation,
) -> SubAgentResult:
    """
    Run a sub-agent as a single bounded LLM call (no full ReAct loop).

    For the v0.3 release this is intentionally minimal: it builds a chat
    message with the role's system_prompt + the request input, calls the
    model at the role's tier, and returns the response text. The tool
    subset is included in the system prompt as a hint but not enforced
    by a separate ReAct loop — that path is reserved for a future
    sub-agent that recursively reuses the parent agent's ReAct machinery.
    """
    role = invocation.role
    parent = invocation.parent_agent

    try:
        model: ModelClient = parent.get_model(role.model_tier)
    except Exception as exc:  # noqa: BLE001
        return SubAgentResult(
            role=request.role, success=False,
            error=f"no model for tier {role.model_tier!r}: {exc}",
        )

    tools_hint = ""
    if role.allowed_tools:
        tools_hint = (
            "\n\nYou may reference these tools by name in your response if "
            "they would be needed: " + ", ".join(role.allowed_tools)
        )
    messages = [
        {"role": "system", "content": role.system_prompt + tools_hint},
        {"role": "user", "content": request.input},
    ]

    response = await model.chat(messages, temperature=0.2, max_tokens=600)
    output = response.content if hasattr(response, "content") else str(response)
    return SubAgentResult(
        role=request.role,
        success=True,
        output=output,
        metadata={"model": model.model, **request.metadata},
    )
