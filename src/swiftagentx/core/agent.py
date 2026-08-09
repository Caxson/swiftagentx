"""
SwiftAgent — the core Agent class.

Provides:
- Dual-model strategy (light for classification, heavy for execution)
- ReAct reasoning loop
- Scenario toolchain shortcuts
- SSE streaming
- Lifecycle hooks for extensibility
- Middleware support
"""

import logging
import re
import time
import uuid
from typing import Any

from ..knowledge_base.base import KnowledgeBase
from ..knowledge_base.tool import KnowledgeBaseTool
from ..middleware.base import Middleware, MiddlewareChain
from ..models.config import ModelTier, SwiftAgentConfig
from ..models.schema import (
    AgentRequest,
    AgentResponse,
    ContextParameters,
    SessionContext,
)
from ..stream.adapter import SSEStreamAdapter
from ..stream.builder import SSEEventBuilder
from ..tools.base import Tool, ToolOutput, ToolOutputType
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolRegistry
from ..tools.scenario import ScenarioCheckpoint, ScenarioConfig, ScenarioEngine
from ..tools.termination import TerminationChecker
from ..tools.workspace_tool import WorkspaceReadTool
from .cache import CacheManager
from .context_offload import offload_if_large, offload_key, truncate_inline
from .hooks import HookContext, HookEvent, HookRegistry, HookResult
from .memory_hooks import TopicChangeHook
from .miner import ReactTranscript, TranscriptMiner
from .memory_layers import (
    InMemoryBackend,
    LayeredMemoryConfig,
    LayeredMemoryStore,
    _default_summarizer_factory,
)
from .model_client import ModelClient
from .parameter import ParameterManager
from .pipeline import RequestPipeline
from .planner import Planner, PlanStore
from .prompt import PromptManager
from .retrieval import ScenarioRetriever
from .router import IntentLevel, IntentResult, IntentRouter
from .skills import Skill, SkillRegistry
from .subagent import (
    SubAgentHandler,
    SubAgentManager,
    SubAgentRequest,
    SubAgentResult,
    SubAgentRole,
)
from .workspace import (
    LocalDiskWorkspaceBackend,
    WorkspaceBackend,
    use_workspace,
)

logger = logging.getLogger(__name__)


class Agent:
    """
    SwiftAgent — enterprise-grade fast-response Agent framework.

    Usage:
        agent = Agent(
            name="MyAgent",
            model=OpenAICompatibleProvider(api_key="...", model="gpt-4"),
        )
        agent.register_tool(MyTool())
        response = await agent.run("What's the weather?")
    """

    def __init__(
        self,
        name: str = "SwiftAgent",
        models: dict[ModelTier, ModelClient] | None = None,
        model: ModelClient | None = None,
        max_iterations: int = 10,
        config: SwiftAgentConfig | None = None,
        scenario_retriever: ScenarioRetriever | None = None,
    ):
        self.name = name
        self.config = config or SwiftAgentConfig(name=name, max_iterations=max_iterations)

        # Model setup
        self._models: dict[ModelTier, ModelClient] = models or {}
        if model is not None:
            if ModelTier.HEAVY not in self._models:
                self._models[ModelTier.HEAVY] = model
            if ModelTier.LIGHT not in self._models:
                self._models[ModelTier.LIGHT] = model

        # Subsystems
        self.memory = LayeredMemoryStore(
            backend=InMemoryBackend(),
            config=LayeredMemoryConfig(
                l2_size=self.config.memory_l2_size,
                l3_max_size=self.config.memory_l3_max_size,
                summarize_every_n_turns=self.config.memory_summarize_every_n_turns,
                summarize_in_background=self.config.memory_summarize_in_background,
            ),
        )
        # Bind the summarizer lazily so the agent's light_model wins even when
        # callers swap models after construction.
        self.memory.set_summarizer_factory(
            lambda: _default_summarizer_factory(
                self.light_model, self.memory.config.summarize_prompt_template,
            )
            if self._models
            else None
        )
        self.cache = CacheManager()
        self.prompt_manager = PromptManager()
        self.parameter_manager = ParameterManager()
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.scenario_engine = ScenarioEngine()
        self.router = IntentRouter(
            retriever=scenario_retriever,
            prefilter_top_k=self.config.scenario_prefilter_top_k,
        )
        self.planner = Planner()
        self.plan_store = PlanStore(
            promote_after=self.config.plan_promote_after,
            auto_reuse=self.config.plan_auto_reuse,
        )
        self.transcript_miner = TranscriptMiner(
            min_cluster_size=self.config.mining_min_cluster_size,
            max_steps=self.planner.max_steps,
        )
        self._react_transcripts: list[ReactTranscript] = []
        self.pipeline = RequestPipeline()
        self.termination_checker = TerminationChecker()
        self.hooks = HookRegistry()

        # Knowledge base
        self._knowledge_base: KnowledgeBase | None = None

        # Middleware
        self._middleware_chain = MiddlewareChain()

        # Max iterations
        self.max_iterations = self.config.max_iterations

        # Default per-instance session id. When run() / run_stream() is called
        # without an explicit session_id, this stable id is used so that
        # successive turns share the same LayeredMemory — the natural
        # behavior for single-user CLI / notebook usage. Multi-user servers
        # should pass an explicit session_id per user.
        self._default_session_id = f"default-{uuid.uuid4().hex[:8]}"

        # Built-in hooks (opt-out via config flags).
        if self.config.memory_enable_topic_change_hook:
            self.hooks.register(TopicChangeHook())

        # MCP clients spun up via register_mcp_server() (kept alive here so
        # they aren't GC'd until shutdown_mcp_servers()).
        self._mcp_clients: dict[str, list[Any]] = {}

        # Sub-agent role registry.
        self.subagents = SubAgentManager()

        # Skill registry — markdown-defined workflows the ReAct loop can invoke.
        self.skills = SkillRegistry()

        # Workspace backend — defaults to local-disk under the system temp
        # dir. Override via agent.workspace_backend = ... before first use.
        self.workspace_backend: WorkspaceBackend = LocalDiskWorkspaceBackend()

        # Read-back side of context offload (D3): large tool results get
        # written to the session workspace instead of inlined into the
        # LLM-facing context (see _execute_react_loop / _execute_scenario);
        # this tool lets the model re-read them on demand. Resolves
        # `self.workspace_backend` lazily so a post-construction override
        # ("before first use") is honored.
        self.tool_registry.register(WorkspaceReadTool(
            lambda: self.workspace_backend,
            max_chars=self.config.context_offload_read_chunk_chars,
        ))

    # --- Model access ---

    def get_model(self, tier: ModelTier = ModelTier.HEAVY) -> ModelClient:
        if tier in self._models:
            return self._models[tier]
        # Fallback: try any available model
        if self._models:
            return next(iter(self._models.values()))
        raise ValueError(f"No model configured for tier '{tier.value}'. Set model in Agent constructor.")

    @property
    def light_model(self) -> ModelClient:
        return self.get_model(ModelTier.LIGHT)

    @property
    def heavy_model(self) -> ModelClient:
        return self.get_model(ModelTier.HEAVY)

    # --- Public API ---

    def set_knowledge_base(
        self,
        kb: KnowledgeBase,
        *,
        auto_short_circuit: bool = True,
        short_circuit_threshold: float | None = None,
    ) -> None:
        """
        Attach a knowledge base to this agent.

        Side effects (in order):

        1. Registers a ``KnowledgeBaseTool`` so the agent can query the KB
           during ReAct loops.
        2. If ``auto_short_circuit`` is True (the default), prepends a
           ``KnowledgeBaseStage`` to the request pipeline so that high-confidence
           exact matches short-circuit the agent and return immediately, with
           zero LLM calls. Pass ``auto_short_circuit=False`` to opt out.

        Args:
            kb: KnowledgeBase implementation to attach.
            auto_short_circuit: Auto-add a ``KnowledgeBaseStage`` to the
                pipeline for sub-second responses on exact matches.
            short_circuit_threshold: Score threshold for the auto-added
                stage. Defaults to ``config.kb_exact_match_threshold``.
        """
        from ..knowledge_base.stage import KnowledgeBaseStage

        self._knowledge_base = kb
        kb_tool = KnowledgeBaseTool(
            kb=kb,
            exact_match_threshold=self.config.kb_exact_match_threshold,
        )
        # Idempotent: if a KB tool is already registered, replace it.
        self.tool_registry.unregister(kb_tool.name)
        self.tool_registry.register(kb_tool)

        if auto_short_circuit:
            threshold = (
                short_circuit_threshold
                if short_circuit_threshold is not None
                else self.config.kb_exact_match_threshold
            )
            # Remove any previously-added auto stage so set_knowledge_base
            # is idempotent.
            self.pipeline.remove_stage("KnowledgeBaseStage")
            self.pipeline.insert_stage(0, KnowledgeBaseStage(kb=kb, threshold=threshold))

    @property
    def knowledge_base(self) -> KnowledgeBase | None:
        return self._knowledge_base

    def register_tool(self, tool: Tool) -> None:
        self.tool_registry.register(tool)

    # ------------------------------------------------------------------
    # MCP server integration
    # ------------------------------------------------------------------

    async def register_mcp_server(
        self,
        name: str,
        *,
        transport: str = "stdio",
        command: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        initialize_timeout: float = 10.0,
        call_timeout: float = 30.0,
    ) -> list[str]:
        """
        Bring up an MCP server, discover its tools, and register them.

        The returned list contains the namespaced tool names (
        ``{name}.{tool}``) that are now in the agent's tool registry.
        Scenarios and the ReAct loop can use them transparently.

        Args:
            name: Server identifier; becomes the namespace prefix for
                tool names (``"postgres"`` → ``"postgres.query"``).
            transport: ``"stdio"`` (subprocess) or ``"sse"`` (HTTP+SSE).
            command: Argv for stdio transport. Required when
                ``transport="stdio"``.
            url: Base URL for sse transport.
            env: Extra env vars for stdio subprocess.
            initialize_timeout: Seconds to wait for the MCP handshake.
            call_timeout: Default per-call timeout for ``tools/call``.

        Returns:
            List of registered (namespaced) tool names.

        Raises:
            MCPClientError: server unavailable, command missing, handshake
                timed out, or transport not supported.
        """
        from ..providers.mcp import MCPClient, MCPServerSpec, MCPTool  # type: ignore

        spec = MCPServerSpec(
            name=name,
            transport=transport,  # type: ignore[arg-type]
            command=command or [],
            url=url,
            env=env or {},
            initialize_timeout=initialize_timeout,
            call_timeout=call_timeout,
        )

        client = MCPClient(spec)
        await client.start()
        try:
            descriptors = await client.list_tools()
        except Exception:
            await client.close()
            raise

        registered: list[str] = []
        for descriptor in descriptors:
            tool = MCPTool(client=client, descriptor=descriptor)
            # Idempotent: replace if a tool with the same qualified name exists.
            self.tool_registry.unregister(tool.name)
            self.tool_registry.register(tool)
            registered.append(tool.name)

        # Keep the client alive on the agent so it's not GC'd; track for shutdown.
        self._mcp_clients.setdefault(name, []).append(client)
        logger.info(
            "Registered %d tools from MCP server %s: %s",
            len(registered), name, registered,
        )
        return registered

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def workspace(
        self,
        session_id: str,
        *,
        cleanup_on_exit: bool = False,
    ) -> Any:
        """Open the per-session workspace as an async context manager.

        Usage::

            async with agent.workspace(session_id="sid") as ws:
                await ws.write("report.pdf", pdf_bytes)
        """
        return use_workspace(
            self.workspace_backend, session_id, cleanup_on_exit=cleanup_on_exit,
        )

    async def _context_safe(self, value: Any, session_id: str, *, key_prefix: str) -> str:
        """Context offload (D3): return ``value`` unchanged when it's small,
        otherwise write it to the session workspace and return a preview +
        reference. Shared by the ReAct loop, the Planner fast path and
        Scenario result formatting so none of them blows the prompt budget
        on one large tool result. Skips opening the workspace entirely when
        offloading wouldn't trigger.

        Offloading is an optimization, so it never fails the request: if the
        workspace can't be written the result degrades to a truncated inline
        copy, which is still bounded.
        """
        threshold = self.config.context_offload_threshold
        text = value if isinstance(value, str) else str(value)
        if threshold <= 0 or len(text) <= threshold:
            return text

        try:
            ws = await self.workspace_backend.open(session_id)
            try:
                return await offload_if_large(
                    text, workspace=ws, key=offload_key(key_prefix),
                    threshold=threshold,
                    preview_chars=self.config.context_offload_preview_chars,
                )
            finally:
                await ws.close()
        except Exception as e:
            logger.warning(
                f"Context offload failed ({e}); truncating {len(text)} chars inline."
            )
            return truncate_inline(text, threshold)

    # ------------------------------------------------------------------
    # Skill registration
    # ------------------------------------------------------------------

    def register_skill(self, skill: Skill) -> None:
        """Register a markdown-defined skill for ReAct invocation."""
        self.skills.register(skill)

    def load_skills(self, directory: str | Any) -> list[str]:
        """Load every ``*.md`` skill under ``directory`` (recursive).

        Returns the names of the skills successfully registered.
        """
        return self.skills.load_dir(directory)

    async def invoke_skill(
        self,
        name: str,
        *,
        args: dict[str, Any] | None = None,
        context_input: str = "",
    ) -> str:
        """
        Run a skill: render its body into a prompt and let the HEAVY (or
        configured-tier) model execute the instructions in one shot.

        Returns the model's textual response. Caller decides what to do
        with it (a ReAct iteration treats it as an observation).
        """
        skill = self.skills.get(name)
        if skill is None:
            return f"[skill {name!r} not found]"

        tier = ModelTier.HEAVY if skill.model_tier == "heavy" else ModelTier.LIGHT
        model = self.get_model(tier)

        rendered_args = ""
        if args:
            rendered_args = "\n".join(f"- {k}: {v}" for k, v in args.items())
            rendered_args = f"\n\nInputs:\n{rendered_args}"

        prompt = (
            f"{skill.to_prompt_block()}\n\n"
            f"Context: {context_input}{rendered_args}\n\n"
            "Execute the skill above step by step and produce the final result."
        )

        response = await model.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        return response.content

    # ------------------------------------------------------------------
    # Sub-agent dispatch
    # ------------------------------------------------------------------

    def register_subagent(
        self, role: SubAgentRole, handler: SubAgentHandler | None = None,
    ) -> None:
        """Register a sub-agent role for use with :meth:`dispatch_subagents`."""
        self.subagents.register_role(role, handler)

    async def dispatch_subagents(
        self,
        requests: list[SubAgentRequest],
        *,
        timeout_seconds: float | None = None,
    ) -> list[SubAgentResult]:
        """
        Run sub-agents in parallel. One :class:`SubAgentResult` per request,
        in the same order. Failed sub-agents return ``success=False``;
        they do not raise.
        """
        return await self.subagents.dispatch(
            self, requests, timeout_seconds=timeout_seconds,
        )

    async def shutdown_mcp_servers(self) -> None:
        """Close every MCP client the agent has spun up."""
        for name, clients in list(self._mcp_clients.items()):
            for client in clients:
                try:
                    await client.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Closing MCP %s raised %s", name, exc)
        self._mcp_clients.clear()

    # ------------------------------------------------------------------
    # Hook dispatch helpers
    # ------------------------------------------------------------------

    async def _dispatch_hook(
        self,
        event: HookEvent,
        *,
        user_input: str,
        session_id: str,
        user_id: str,
        memory: Any = None,
        **fields: Any,
    ) -> HookResult:
        """Build a HookContext and run every registered hook for ``event``."""
        if not self.hooks.list_hooks(event):
            return HookResult()
        ctx = HookContext(
            event=event,
            user_input=user_input,
            session_id=session_id,
            user_id=user_id,
            agent=self,
            memory=memory,
            **fields,
        )
        return await self.hooks.dispatch(event, ctx)

    def register_scenario(self, scenario_id: str, scenario: ScenarioConfig) -> None:
        self.scenario_engine.register(scenario_id, scenario)
        # Hand the scenario's required template slots to the router so the
        # classifier can extract them from natural language in the same
        # LLM call — otherwise a Scenario like weather(city=$city) can only
        # fire when the caller pre-parses `city` and passes it as a kwarg.
        self.router.register_scenarios({scenario_id: {
            "name": scenario.name,
            "description": scenario.description,
            # Triggers are retrieval anchors for the prefilter, not prompt
            # content — they never inflate the classification prompt.
            "triggers": scenario.triggers,
            "slots": sorted(scenario.required_vars()),
        }})

    def use(self, middleware: Middleware) -> None:
        self._middleware_chain.add(middleware)

    def _sanitize_error(self, error: Exception) -> str:
        """Return a safe error message. Only expose details when debug=True."""
        if self.config.debug:
            return f"Error: {error}"
        return "Sorry, an internal error occurred. Please try again later."

    def _validate_input(self, user_input: str) -> None:
        """Validate user input length."""
        if len(user_input) > self.config.max_input_length:
            raise ValueError(
                f"Input exceeds maximum length ({self.config.max_input_length} characters)"
            )

    async def run(self, user_input: Any, **context_vars: Any) -> AgentResponse:
        """
        Process a user request (non-streaming).

        Accepts either:

        - a plain string ``user_input`` plus optional ``session_id`` /
          ``user_id`` kwargs (the common convenience shape used in the
          README quick-start examples), or
        - an :class:`AgentRequest` instance (the same shape used by
          :meth:`run_stream`) — its fields are unpacked into
          ``context_vars`` automatically.

        Mixing the two is an error: passing an ``AgentRequest`` *and*
        kwargs at the same time raises a clear ``TypeError`` rather than
        silently dropping the kwargs.

        Returns:
            AgentResponse with the final answer.
        """
        # Accept AgentRequest as a polymorphic alternative to (str, **kw).
        # Without this, ``agent.run(AgentRequest(...))`` — a reasonable
        # user expectation given run_stream takes one — crashes deep
        # inside _validate_input with a confusing
        # ``object of type 'AgentRequest' has no len()`` TypeError.
        if isinstance(user_input, AgentRequest):
            if context_vars:
                raise TypeError(
                    "Agent.run() received an AgentRequest plus extra "
                    "keyword arguments — pick one calling style: either "
                    "`run(request)` or `run(text, session_id=..., user_id=...)`."
                )
            req = user_input
            user_input = req.user_input
            context_vars = {
                "session_id": req.session_id,
                "user_id": req.user_id,
                "app_version": req.app_version,
                "platform": req.platform,
                "device_id": req.device_id,
                "channel": req.channel,
                **(req.extra_params or {}),
            }

        request_id = str(uuid.uuid4())
        start_time = time.time()
        session_id = context_vars.get("session_id") or self._default_session_id
        user_id = context_vars.get("user_id", "anonymous")

        # Convert input-validation failures into an AgentResponse instead
        # of raising ValueError out of run(). A web handler that didn't
        # wrap run() in try/except would otherwise return a 500 with a
        # leaky stack trace just because the user sent a long message.
        try:
            self._validate_input(user_input)
        except ValueError as exc:
            return AgentResponse(
                session_id=session_id,
                request_id=request_id,
                answer=str(exc),
                total_iterations=0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"input_rejected": True, "error_class": "ValueError"},
            )

        # If any user-registered middleware exists, wrap the rest of run()
        # inside the chain so middlewares can log before/after, mutate the
        # request, or short-circuit. The chain calls _inner_run() as the
        # innermost handler. (Dogfood Friction #9: prior to v0.3.x the
        # middleware chain was constructed and `agent.use()` appended into
        # it, but Agent.run() never executed the chain.)
        if self._middleware_chain._middlewares:
            mw_ctx: dict[str, Any] = {
                "request_id": request_id,
                "session_id": session_id,
                "user_id": user_id,
                "user_input": user_input,
                **context_vars,
            }

            async def _inner_handler(_ctx: dict[str, Any]) -> dict[str, Any]:
                resp = await self._run_internal(
                    user_input=user_input, request_id=request_id,
                    start_time=start_time, session_id=session_id,
                    user_id=user_id, context_vars=context_vars,
                )
                _ctx["response"] = resp
                return _ctx

            final_ctx = await self._middleware_chain.execute(mw_ctx, inner=_inner_handler)
            return final_ctx["response"]

        return await self._run_internal(
            user_input=user_input, request_id=request_id,
            start_time=start_time, session_id=session_id,
            user_id=user_id, context_vars=context_vars,
        )

    async def _run_internal(
        self,
        *,
        user_input: str,
        request_id: str,
        start_time: float,
        session_id: str,
        user_id: str,
        context_vars: dict[str, Any],
    ) -> AgentResponse:
        """Internal body of run() that the middleware chain wraps."""

        # Anything the caller passed as a non-reserved kwarg becomes a
        # context variable so Scenario tool_chain's $foo templates and
        # tools that look at context.get(...) can see it. The reserved
        # keys (session_id, user_id, platform) stay at the top level of
        # context_vars and DON'T get duplicated into variables.
        _reserved = {"session_id", "user_id", "platform"}
        scenario_vars: dict[str, Any] = {
            k: v for k, v in context_vars.items() if k not in _reserved
        }
        context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            max_iterations=self.max_iterations,
            variables=scenario_vars,
        )

        # Acquire the per-session LayeredMemory (lazy-creates on first turn).
        mem = await self.memory.get(session_id, user_id)
        is_first_turn = mem.total_turns_added == 0

        try:
            # Lifecycle: subclass method (legacy) + Hook system (declarative).
            await self.on_request_start(context)
            if is_first_turn:
                await self._dispatch_hook(
                    HookEvent.SESSION_START,
                    user_input=user_input, session_id=session_id,
                    user_id=user_id, memory=mem,
                )
            await self._dispatch_hook(
                HookEvent.REQUEST_START,
                user_input=user_input, session_id=session_id,
                user_id=user_id, memory=mem,
            )

            # Run request pipeline (KB short-circuit, security checks, etc.).
            # A stage that returns SHORT_CIRCUIT bypasses the rest of run().
            if self.pipeline.stages:
                pipeline_ctx: dict[str, Any] = {
                    "user_input": user_input,
                    "user_id": user_id,
                    "session_id": session_id,
                    **context_vars,
                }
                from .pipeline import StageAction
                stage_result = await self.pipeline.execute(pipeline_ctx)
                if stage_result.action == StageAction.SHORT_CIRCUIT and stage_result.answer is not None:
                    answer = stage_result.answer
                    stored_answer = await self._context_safe(
                        answer, session_id, key_prefix="memory_turn",
                    )
                    await mem.add_turn(user_input, stored_answer)
                    return AgentResponse(
                        session_id=session_id,
                        request_id=request_id,
                        answer=answer,
                        total_iterations=0,
                        execution_time_ms=(time.time() - start_time) * 1000,
                        metadata={
                            "pipeline_short_circuit": True,
                            **stage_result.metadata,
                        },
                    )
                if stage_result.action == StageAction.ABORT:
                    answer = stage_result.answer or "Request aborted by pipeline."
                    return AgentResponse(
                        session_id=session_id,
                        request_id=request_id,
                        answer=answer,
                        total_iterations=0,
                        execution_time_ms=(time.time() - start_time) * 1000,
                        metadata={"pipeline_abort": True, **stage_result.metadata},
                    )

            # Cache check
            if self.config.enable_cache:
                hit, cached_value, source = self.cache.query(
                    user_input, user_id, session_id,
                    platform=context_vars.get("platform", "default"),
                )
                if hit:
                    answer = str(cached_value)
                    stored_answer = await self._context_safe(
                        answer, session_id, key_prefix="memory_turn",
                    )
                    await mem.add_turn(user_input, stored_answer)
                    return AgentResponse(
                        session_id=session_id,
                        request_id=request_id,
                        answer=answer,
                        total_iterations=0,
                        execution_time_ms=(time.time() - start_time) * 1000,
                        metadata={"cache_hit": True, "cache_source": source},
                    )

            # Intent classification (BEFORE_CLASSIFY is where TopicChangeHook
            # detects a topic change and rolls L3 into L4 before downstream
            # rendering sees the memory).
            await self.on_before_classify(context)
            hook_result = await self._dispatch_hook(
                HookEvent.BEFORE_CLASSIFY,
                user_input=user_input, session_id=session_id,
                user_id=user_id, memory=mem,
            )
            if hook_result.action == "short_circuit" and hook_result.answer is not None:
                answer = hook_result.answer
                stored_answer = await self._context_safe(
                    answer, session_id, key_prefix="memory_turn",
                )
                await mem.add_turn(user_input, stored_answer)
                return AgentResponse(
                    session_id=session_id, request_id=request_id, answer=answer,
                    total_iterations=0,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    metadata={"hook_short_circuit": True, **hook_result.metadata},
                )

            intent = await self._classify_intent(user_input, context)
            await self.on_after_classify(context, intent)
            await self._dispatch_hook(
                HookEvent.AFTER_CLASSIFY,
                user_input=user_input, session_id=session_id,
                user_id=user_id, memory=mem, intent=intent,
                scenario=intent.scenario,
            )

            # Execute based on intent. Merge any classifier-extracted
            # slots into context vars and downgrade SCENARIO→REACT if a
            # required template slot is still missing.
            exec_level = self._resolve_execution_level(intent, context)
            if exec_level == IntentLevel.SCENARIO and intent.scenario:
                answer = await self._execute_scenario(intent.scenario, context)
            elif exec_level == IntentLevel.REACT:
                answer = None
                if self.config.enable_planner:
                    answer = await self._try_planned_execution(context)
                if answer is None:
                    answer = await self._react_loop(context)
            else:
                answer = await self._direct_response(context)

            # Post-processing hook
            answer = await self.on_before_respond(context, answer)
            await self._dispatch_hook(
                HookEvent.BEFORE_RESPOND,
                user_input=user_input, session_id=session_id,
                user_id=user_id, memory=mem, answer=answer,
            )

            # Record the completed turn (folds into L2; may fire summarize).
            # A direct-output answer is returned to the user verbatim (`answer`
            # below is untouched), but the copy that lands in L2 gets the same
            # offload treatment as D3's per-call tool results (D3b) — otherwise
            # it gets replayed whole into every later prompt via to_chat_messages.
            stored_answer = await self._context_safe(
                answer, session_id, key_prefix="memory_turn",
            )
            await mem.add_turn(user_input, stored_answer)

            # Populate the L2 cache with this answer so the next identical
            # (user_input, user_id, platform) request inside the TTL window
            # short-circuits at the cache.query() check at the top of run().
            # Prior to this commit, CacheManager.set_level_2 / set_scenario_cache
            # / set_level_1 / set_level_3 were declared and tested as classes
            # but the agent never WROTE to any of them — so cache.query()
            # always returned (False, None, "") and the "three-level cache"
            # headline of v0.3 was effectively a no-op.
            if self.config.enable_cache:
                self.cache.set_level_2(
                    user_input, user_id, answer,
                    ttl_seconds=self.config.code_cache_ttl,
                    platform=context_vars.get("platform", "default"),
                )

            elapsed_ms = (time.time() - start_time) * 1000
            response = AgentResponse(
                session_id=session_id,
                request_id=request_id,
                answer=answer,
                total_iterations=context.current_iteration,
                execution_time_ms=elapsed_ms,
                metadata={
                    "intent_level": intent.level.value,
                    "scenario": intent.scenario,
                    "cached_for_next": self.config.enable_cache,
                },
            )

            await self.on_request_end(context, response)
            await self._dispatch_hook(
                HookEvent.REQUEST_END,
                user_input=user_input, session_id=session_id,
                user_id=user_id, memory=mem, answer=answer,
            )
            return response

        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)
            elapsed_ms = (time.time() - start_time) * 1000
            # debug=False (production default): expose only the exception
            # CLASS so callers can branch on the failure mode without
            # leaking the raw message — which might contain secrets
            # (DB URLs, API keys, etc.). debug=True: full message + traceback.
            metadata: dict[str, Any] = {
                "error_class": type(e).__name__,
            }
            if self.config.debug:
                import traceback as _tb
                metadata["error_message"] = str(e)[:240]
                metadata["traceback"] = _tb.format_exc()
                metadata["error"] = str(e)
            return AgentResponse(
                session_id=session_id,
                request_id=request_id,
                answer=self._sanitize_error(e),
                total_iterations=context.current_iteration,
                execution_time_ms=elapsed_ms,
                metadata=metadata,
            )

    async def run_stream(
        self, request: AgentRequest, adapter: SSEStreamAdapter
    ) -> AgentResponse:
        """
        Process a user request with SSE streaming.

        Args:
            request: AgentRequest
            adapter: SSEStreamAdapter for sending events

        Returns:
            AgentResponse
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        try:
            self._validate_input(request.user_input)
        except ValueError as exc:
            return AgentResponse(
                session_id=request.session_id,
                request_id=request_id,
                answer=str(exc),
                total_iterations=0,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"input_rejected": True, "error_class": "ValueError"},
            )

        context = SessionContext(
            session_id=request.session_id,
            user_id=request.user_id,
            user_input=request.user_input,
            max_iterations=self.max_iterations,
            parameters=ContextParameters(
                app_version=request.app_version,
                platform=request.platform,
                device_id=request.device_id,
                channel=request.channel,
                extra_params=request.extra_params,
            ),
        )

        mem = await self.memory.get(request.session_id, request.user_id)
        is_first_turn = mem.total_turns_added == 0

        try:
            await self.on_request_start(context)
            if is_first_turn:
                await self._dispatch_hook(
                    HookEvent.SESSION_START,
                    user_input=request.user_input, session_id=request.session_id,
                    user_id=request.user_id, memory=mem,
                )
            await self._dispatch_hook(
                HookEvent.REQUEST_START,
                user_input=request.user_input, session_id=request.session_id,
                user_id=request.user_id, memory=mem,
            )
            await adapter.send_event(SSEEventBuilder.initialized(f"{self.name} initialized"))

            # Cache check
            if self.config.enable_cache:
                hit, cached_value, source = self.cache.query(
                    request.user_input, request.user_id, request.session_id,
                    platform=request.platform,
                )
                if hit:
                    await adapter.send_event(SSEEventBuilder.cache_hit(source, str(cached_value)))
                    answer = str(cached_value)
                    await self._stream_answer(answer, adapter, request.session_id, request_id)
                    await adapter.finish()
                    stored_answer = await self._context_safe(
                        answer, request.session_id, key_prefix="memory_turn",
                    )
                    await mem.add_turn(request.user_input, stored_answer)
                    return AgentResponse(
                        session_id=request.session_id, request_id=request_id,
                        answer=answer, total_iterations=0,
                        execution_time_ms=(time.time() - start_time) * 1000,
                        metadata={"cache_hit": True, "cache_source": source},
                    )

            # Classification (BEFORE_CLASSIFY hook can fire TopicChange + summarize).
            await self.on_before_classify(context)
            await self._dispatch_hook(
                HookEvent.BEFORE_CLASSIFY,
                user_input=request.user_input, session_id=request.session_id,
                user_id=request.user_id, memory=mem,
            )
            intent = await self._classify_intent(request.user_input, context)
            await self.on_after_classify(context, intent)
            await self._dispatch_hook(
                HookEvent.AFTER_CLASSIFY,
                user_input=request.user_input, session_id=request.session_id,
                user_id=request.user_id, memory=mem, intent=intent,
                scenario=intent.scenario,
            )

            # Execute. Track whether the chosen path already streamed the
            # answer to the adapter; if so, we must NOT re-emit it via
            # _stream_answer or the client receives the answer twice
            # (dogfood Friction #8).
            answer_already_streamed = False
            exec_level = self._resolve_execution_level(intent, context)
            if exec_level == IntentLevel.SCENARIO and intent.scenario:
                answer = await self._execute_scenario(intent.scenario, context)
            elif exec_level == IntentLevel.REACT:
                answer = None
                if self.config.enable_planner:
                    # Planned execution doesn't stream per-step events; the
                    # final answer still goes out via _stream_answer below.
                    answer = await self._try_planned_execution(context)
                if answer is None:
                    answer = await self._react_loop(context, adapter)
                # _react_loop streams thoughts/actions/observations but
                # currently does NOT stream the final answer token-by-token,
                # so we still need _stream_answer to emit it.
            else:
                answer = await self._direct_response(context, adapter)
                answer_already_streamed = True  # _direct_response streamed chunks

            answer = await self.on_before_respond(context, answer)
            await self._dispatch_hook(
                HookEvent.BEFORE_RESPOND,
                user_input=request.user_input, session_id=request.session_id,
                user_id=request.user_id, memory=mem, answer=answer,
            )
            if answer_already_streamed:
                # The chunks are already out; just close with answer_end so
                # clients know the answer body is complete.
                await adapter.send_event(SSEEventBuilder.answer_end(answer))
            else:
                await self._stream_answer(answer, adapter, request.session_id, request_id)
            await adapter.send_event(SSEEventBuilder.completed())
            await adapter.finish()

            stored_answer = await self._context_safe(
                answer, request.session_id, key_prefix="memory_turn",
            )
            await mem.add_turn(request.user_input, stored_answer)
            if self.config.enable_cache:
                self.cache.set_level_2(
                    request.user_input, request.user_id, answer,
                    ttl_seconds=self.config.code_cache_ttl,
                    platform=request.platform,
                )
            elapsed_ms = (time.time() - start_time) * 1000

            response = AgentResponse(
                session_id=request.session_id, request_id=request_id,
                answer=answer, total_iterations=context.current_iteration,
                execution_time_ms=elapsed_ms,
                metadata={
                    "intent_level": intent.level.value,
                    "scenario": intent.scenario,
                    "cached_for_next": self.config.enable_cache,
                },
            )
            await self.on_request_end(context, response)
            await self._dispatch_hook(
                HookEvent.REQUEST_END,
                user_input=request.user_input, session_id=request.session_id,
                user_id=request.user_id, memory=mem, answer=answer,
            )
            return response

        except Exception as e:
            logger.error(f"Agent stream run failed: {e}", exc_info=True)
            error_msg = self._sanitize_error(e)
            # See note in run(): debug=False must NOT leak the raw exception
            # message — it can contain secrets the framework didn't sanitise.
            try:
                await adapter.send_event(SSEEventBuilder.error(error_msg))
                await adapter.finish()
            except Exception:
                # Adapter may be already closed (e.g. client disconnected) —
                # don't let that mask the real error.
                pass
            metadata: dict[str, Any] = {
                "error_class": type(e).__name__,
            }
            if self.config.debug:
                import traceback as _tb
                metadata["error_message"] = str(e)[:240]
                metadata["traceback"] = _tb.format_exc()
                metadata["error"] = str(e)
            return AgentResponse(
                session_id=request.session_id, request_id=request_id,
                answer=error_msg, total_iterations=context.current_iteration,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata=metadata,
            )

    # --- Lifecycle hooks (override in subclass) ---

    async def on_request_start(self, context: SessionContext) -> None:
        """Called at the beginning of request processing."""
        pass

    async def on_before_classify(self, context: SessionContext) -> None:
        """Called before intent classification."""
        pass

    async def on_after_classify(self, context: SessionContext, result: IntentResult) -> None:
        """Called after intent classification."""
        pass

    async def on_before_tool_call(self, context: SessionContext, tool_name: str, params: dict[str, Any]) -> None:
        """Called before each tool call."""
        pass

    async def on_after_tool_call(self, context: SessionContext, tool_name: str, result: ToolOutput) -> None:
        """Called after each tool call."""
        pass

    async def on_before_respond(self, context: SessionContext, answer: str) -> str:
        """Called before sending the final answer. Return modified answer."""
        return answer

    async def on_request_end(self, context: SessionContext, response: AgentResponse) -> None:
        """Called at the end of request processing."""
        pass

    # --- Internal methods ---

    async def _classify_intent(self, user_input: str, context: SessionContext) -> IntentResult:
        """Classify user intent using the light model."""
        try:
            model = self.get_model(ModelTier.LIGHT)
        except ValueError:
            return IntentResult(level=IntentLevel.DIRECT, confidence=0.5, raw_output="no light model")

        return await self.router.classify(user_input, model=model)

    async def _react_loop(
        self, context: SessionContext, adapter: SSEStreamAdapter | None = None
    ) -> str:
        """Execute the ReAct reasoning loop."""
        model = self.get_model(ModelTier.HEAVY)
        accumulated_context = ""
        answer = ""
        executed_actions: list[tuple[str, dict[str, Any]]] = []

        # Per-loop helper for dispatching iteration-scoped hooks.
        async def _dispatch_iter(event: HookEvent, **extra: Any) -> None:
            mem = await self.memory.get(context.session_id, context.user_id)
            await self._dispatch_hook(
                event,
                user_input=context.user_input,
                session_id=context.session_id,
                user_id=context.user_id,
                memory=mem,
                **extra,
            )

        for iteration in range(self.max_iterations):
            await _dispatch_iter(HookEvent.BEFORE_REACT_ITER, react_iteration=iteration + 1)
            context.current_iteration = iteration + 1

            if adapter:
                await adapter.send_event(SSEEventBuilder.thought_start(iteration + 1))

            # Generate thought
            thought = await self._generate_thought(context, model, accumulated_context)

            if adapter:
                await adapter.send_event(SSEEventBuilder.thought_end(thought, iteration + 1))

            context.add_step("THOUGHT", thought)

            # Check termination. We enable check_repeated_actions by default
            # so a ReAct loop that decides to call the same tool with the
            # same args twice in a row stops instead of burning tokens for
            # several more iterations (dogfood Friction #6).
            should_stop, reason = self.termination_checker.should_terminate(
                context, thought,
                check_repeated_actions=True,
            )
            if should_stop:
                logger.info(f"ReAct loop terminated: {reason}")
                break

            # Parse action from thought
            tool_name, tool_params = self._parse_action(thought)

            if tool_name:
                # Skip executing an action we already executed earlier in
                # this loop — the LLM has the observation already. Without
                # this, models will gladly call the same tool with the same
                # args 5-10 times before producing a final answer
                # (dogfood Friction #6). We break the loop and let the
                # caller synthesise the final answer from the last
                # observation in accumulated_context.
                #
                # Dedup key normalises whitespace inside string params so
                # ``calculator(12*34)`` and ``calculator(12 * 34)`` count
                # as the same call — LLMs love to "retry" with cosmetic
                # variations on the prior call's arg formatting (dogfood
                # Round 5). We strip all whitespace inside string values
                # only; we don't touch the actual tool_params passed to
                # the tool, so semantically-significant whitespace (e.g.,
                # search queries) is preserved when the tool runs.
                def _dedup_key(name: str, params: dict[str, Any]) -> str:
                    norm = {
                        k: ("".join(v.split()) if isinstance(v, str) else v)
                        for k, v in params.items()
                    }
                    return f"{name}({norm})"

                proposed_action = _dedup_key(tool_name, tool_params)
                prior_dedup_keys = [
                    s.get("metadata", {}).get("dedup_key", "") for s in context.steps
                    if s.get("type") == "ACTION"
                ]
                if proposed_action in prior_dedup_keys:
                    logger.info(
                        "ReAct: refusing duplicate action %s; "
                        "asking the model to finalise from existing observations.",
                        proposed_action,
                    )
                    break

                if adapter:
                    await adapter.send_event(SSEEventBuilder.action_start(tool_name, iteration + 1))

                await self.on_before_tool_call(context, tool_name, tool_params)
                await _dispatch_iter(
                    HookEvent.BEFORE_TOOL_CALL,
                    tool_name=tool_name, tool_args=tool_params,
                    react_iteration=iteration + 1,
                )
                result = await self.tool_executor.execute(tool_name, context, **tool_params)
                if result.success:
                    executed_actions.append((tool_name, dict(tool_params)))
                await self.on_after_tool_call(context, tool_name, result)
                await _dispatch_iter(
                    HookEvent.AFTER_TOOL_CALL,
                    tool_name=tool_name, tool_args=tool_params,
                    tool_result=result, react_iteration=iteration + 1,
                )

                observation = str(result.result) if result.success else f"Error: {result.error}"
                tool = self.tool_registry.get(tool_name)
                if tool is not None and getattr(tool, "offload_exempt", False):
                    # `workspace_read` and friends already bound their own
                    # output; offloading it again would mean the model can
                    # never read an offloaded result back.
                    context_observation = observation
                else:
                    context_observation = await self._context_safe(
                        observation, context.session_id,
                        key_prefix=f"react_{tool_name}_{context.current_iteration}",
                    )
                accumulated_context += f"\nTool: {tool_name}\nResult: {context_observation}\n"

                context.add_step(
                    "ACTION",
                    f"{tool_name}({tool_params})",
                    metadata={"dedup_key": proposed_action},
                )
                context.add_step("OBSERVATION", observation)

                if adapter:
                    await adapter.send_event(
                        SSEEventBuilder.action_end(tool_name, tool_params, iteration + 1)
                    )
                    await adapter.send_event(SSEEventBuilder.observation(observation, iteration + 1))

                # Check if the tool returned a direct output
                if result.success and result.output_type == ToolOutputType.DIRECT_OUTPUT:
                    self._record_react_transcript(context, executed_actions)
                    return observation
            else:
                # No action — extract answer from thought
                answer = self._extract_final_answer(thought)
                if answer:
                    await _dispatch_iter(HookEvent.AFTER_REACT_ITER, react_iteration=iteration + 1)
                    self._record_react_transcript(context, executed_actions)
                    return answer

            await _dispatch_iter(HookEvent.AFTER_REACT_ITER, react_iteration=iteration + 1)

        # Generate final answer from accumulated context
        final_answer = await self._generate_final_answer(context, model, accumulated_context)
        self._record_react_transcript(context, executed_actions)
        return final_answer

    async def _generate_thought(
        self, context: SessionContext, model: ModelClient, accumulated_context: str
    ) -> str:
        """Generate a thought using the heavy model."""
        tools_desc = "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in self.tool_registry.get_all().items()
        )

        prompt = (
            f"You are {self.name}. Use the ReAct pattern to solve the user's request.\n\n"
            f"Available tools:\n{tools_desc}\n\n"
            f"Conversation context:\n{accumulated_context}\n\n"
            f"User input: {context.user_input}\n"
            f"Iteration: {context.current_iteration}/{self.max_iterations}\n\n"
            "Think step by step. If you need to call a tool, respond with:\n"
            "Thought: [your reasoning]\n"
            "Action: [tool_name]\n"
            "Action Input: [JSON parameters]\n\n"
            "If you have the final answer, respond with:\n"
            "Final Answer: [your answer]"
        )

        response = await model.chat([{"role": "user", "content": prompt}])
        return response.content

    def _parse_action(self, thought: str) -> tuple[str, dict[str, Any]]:
        """Parse tool name and parameters from a thought."""
        import json as json_mod

        # Match "Action: tool_name"
        action_match = re.search(r"Action:\s*(\w+)", thought)
        if not action_match:
            return "", {}

        tool_name = action_match.group(1)

        # Match "Action Input: {...}"
        input_match = re.search(r"Action Input:\s*(\{.*?\})", thought, re.DOTALL)
        params: dict[str, Any] = {}
        if input_match:
            try:
                params = json_mod.loads(input_match.group(1))
            except json_mod.JSONDecodeError:
                pass

        return tool_name, params

    def _extract_final_answer(self, thought: str) -> str:
        """Extract final answer from thought text."""
        match = re.search(r"Final Answer:\s*(.+)", thought, re.DOTALL)
        return match.group(1).strip() if match else ""

    async def _generate_final_answer(
        self, context: SessionContext, model: ModelClient, accumulated_context: str
    ) -> str:
        """Generate final answer from accumulated context."""
        prompt = (
            f"{self.prompt_manager.get_output_prompt()}\n\n"
            f"Context and tool results:\n{accumulated_context}\n\n"
            f"User's original question: {context.user_input}\n\n"
            "Please provide a concise, helpful answer."
        )

        response = await model.chat([{"role": "user", "content": prompt}])
        return response.content

    def _resolve_execution_level(
        self, intent: IntentResult, context: SessionContext
    ) -> IntentLevel:
        """Merge classifier-extracted slots into context vars, then decide
        the execution path.

        A SCENARIO intent is downgraded to REACT when the matched scenario
        still has required template slots that the classifier couldn't fill
        from the user input — better to spend the extra LLM calls and
        actually answer than to fire a scenario step with an unsubstituted
        ``$city`` and fail. Scenarios with all slots satisfied (or no slots
        at all) keep the fast 1-LLM-call path.
        """
        if intent.level != IntentLevel.SCENARIO or not intent.scenario:
            return intent.level

        for key, value in (intent.slots or {}).items():
            if value:
                context.set_variable(key, value)

        scenario = self.scenario_engine.get(intent.scenario)
        if scenario is not None:
            missing = [
                name for name in scenario.required_vars()
                if not context.get_variable(name)
            ]
            if missing:
                logger.info(
                    "Scenario '%s' missing slot(s) %s after classification — "
                    "falling back to ReAct.", intent.scenario, missing,
                )
                return IntentLevel.REACT
        return IntentLevel.SCENARIO

    async def _execute_scenario(self, scenario_id: str, context: SessionContext) -> str:
        """Execute a pre-defined scenario toolchain.

        ``extra_vars`` is what the scenario engine substitutes into
        ``ToolChainStep.query_template`` and uses to build cache keys.
        It must include the caller's ``agent.run(..., **kwargs)`` so that
        a step like ``ToolChainStep(tool="weather", query_template="$city")``
        actually substitutes the city the caller passed. Reserved keys
        (user_id / user_input / session_id) are merged on top.

        We also wire the engine's per-step callback to dispatch
        ``HookEvent.BEFORE_SCENARIO_STEP`` / ``AFTER_SCENARIO_STEP`` so
        registered hooks fire per-step.
        """
        extra_vars = {
            **dict(context.variables),
            "user_id": context.user_id,
            "user_input": context.user_input,
            "session_id": context.session_id,
        }

        # Scenario-level result cache. Opt-in: only scenarios that declared
        # a `cache_key_template` (or a custom key builder) are cached. The
        # key is semantic (e.g. `order_status_$user_id`), so it dedups
        # across different *phrasings* of the same intent — which the
        # request-level cache (keyed on raw input text) cannot. Previously
        # `cache_key_template`/`cache_ttl` and `build_cache_key()` were dead
        # code: declared and documented but never read (GitHub #1).
        cache_on = self.config.enable_cache and self.scenario_engine.is_cacheable(scenario_id)
        cache_key: str | None = None
        if cache_on:
            cache_key = self.scenario_engine.build_cache_key(scenario_id, extra_vars)
            cached = self.cache.get_scenario_cache(scenario_id, cache_key)
            if cached is not None:
                return str(cached)

        async def _on_step(phase: str, step: Any, tool_kwargs: dict[str, Any],
                           output: Any) -> None:
            event = (HookEvent.BEFORE_SCENARIO_STEP if phase == "before"
                     else HookEvent.AFTER_SCENARIO_STEP)
            mem = await self.memory.get(context.session_id, context.user_id)
            await self._dispatch_hook(
                event,
                user_input=context.user_input,
                session_id=context.session_id,
                user_id=context.user_id,
                memory=mem,
                scenario=scenario_id,
                tool_name=step.tool,
                tool_args=dict(tool_kwargs),
                tool_result=output,
            )

        # D4: checkpoint progress after every step-group so a chain
        # interrupted mid-run (process crash, or a step failure the caller
        # retries after fixing) resumes instead of re-running finished
        # steps. Keyed by (session, scenario) — deterministic, not
        # suffixed like D3's offload keys, since resume must find the same
        # file a later invocation of the same scenario/session writes.
        ws = await self.workspace_backend.open(context.session_id)
        checkpoint = ScenarioCheckpoint(ws, f"scenario_{scenario_id}")

        result = await self.scenario_engine.execute(
            scenario_id, context, self.tool_executor,
            extra_vars=extra_vars,
            step_callback=_on_step,
            checkpoint=checkpoint,
        )

        if result.success:
            if result.output_type == ToolOutputType.DIRECT_OUTPUT:
                answer = str(result.result)
            else:
                # Use heavy model to format the result. Large results are
                # offloaded to the workspace first (D3) so one oversized
                # step doesn't blow the formatting prompt's budget.
                model = self.get_model(ModelTier.HEAVY)
                context_result = await self._context_safe(
                    result.result, context.session_id,
                    key_prefix=f"scenario_{scenario_id}",
                )
                prompt = (
                    f"{self.prompt_manager.get_output_prompt()}\n\n"
                    f"Tool results: {context_result}\n\n"
                    f"User question: {context.user_input}\n\n"
                    "Generate a natural, helpful response."
                )
                response = await model.chat([{"role": "user", "content": prompt}])
                answer = response.content
            # Only successful results are cached, with the scenario's own TTL.
            if cache_on and cache_key is not None:
                scenario = self.scenario_engine.get(scenario_id)
                ttl = scenario.cache_ttl if scenario else 3600
                self.cache.set_scenario_cache(scenario_id, cache_key, answer, ttl_seconds=ttl)
            return answer
        else:
            return f"Sorry, I couldn't complete this request: {result.error}"

    async def _try_planned_execution(self, context: SessionContext) -> str | None:
        """Planner fast path for REACT-level requests. None = fall back to ReAct.

        Order matters: a cached-plan hit costs one slot-extraction call; a
        fresh plan costs one planning call — both beat the N+1 calls of a
        full ReAct loop. Every failure path returns None instead of
        raising, so this can only make requests faster, never break them.
        """
        user_input = context.user_input

        # 1) Reuse a cached (probation) plan when one matches this phrasing.
        cached = self.plan_store.match(user_input)
        if cached is not None:
            slots = await self._extract_plan_slots(cached.slot_names, user_input)
            if slots is not None:
                answer = await self._execute_plan_steps(
                    cached.steps, slots, context, cached.intent,
                )
                if answer is not None:
                    self._after_plan_success(cached.plan_id, user_input, slots)
                    return answer
                self.plan_store.record_failure(cached.plan_id)
                return None
            # Slots unextractable for this phrasing — try fresh planning.

        # 2) Fresh plan: one light-model call.
        try:
            model = self.get_model(ModelTier.LIGHT)
        except ValueError:
            return None
        schemas = self.tool_registry.schemas_for_query(user_input)
        if not schemas:
            return None
        prompt = self.planner.build_prompt(user_input, schemas)
        try:
            response = await model.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=500,
            )
        except Exception as e:
            logger.error(f"Plan generation failed: {e}", exc_info=True)
            return None

        plan = self.planner.parse(response.content)
        if plan is None:
            return None
        error = self.planner.validate(plan, set(self.tool_registry.list_tools()))
        if error is not None:
            logger.info(f"Plan rejected: {error}")
            return None
        if plan.required_vars() - set(plan.slots):
            logger.info("Plan rejected: required slot values missing from request")
            return None

        answer = await self._execute_plan_steps(plan.steps, plan.slots, context, plan.intent)
        if answer is None:
            return None
        # Only plans that actually executed successfully enter the cache.
        cached = self.plan_store.add(plan, user_input)
        self._after_plan_success(cached.plan_id)
        return answer

    async def _execute_plan_steps(
        self,
        steps: list[Any],
        slots: dict[str, str],
        context: SessionContext,
        intent: str,
    ) -> str | None:
        """Run a plan's tool chain deterministically. None = step failed."""
        config = ScenarioConfig(name=intent, tool_chain=list(steps))
        extra_vars = {
            **dict(context.variables),
            **slots,
            "user_id": context.user_id,
            "user_input": context.user_input,
            "session_id": context.session_id,
        }
        result = await self.scenario_engine.execute_config(
            config, f"plan:{intent}", context, self.tool_executor,
            extra_vars=extra_vars,
        )
        if not result.success:
            logger.info(
                f"Plan '{intent}' step failed ({result.error}) — falling back to ReAct"
            )
            return None
        # Tools already ran (side effects included) — from here on, never
        # return None or the ReAct fallback would run them a second time.
        try:
            model = self.get_model(ModelTier.HEAVY)
            context_result = await self._context_safe(
                result.result, context.session_id, key_prefix=f"plan_{intent}",
            )
            prompt = (
                f"{self.prompt_manager.get_output_prompt()}\n\n"
                f"Tool results: {context_result}\n\n"
                f"User question: {context.user_input}\n\n"
                "Generate a natural, helpful response."
            )
            response = await model.chat([{"role": "user", "content": prompt}])
            return response.content
        except Exception as e:
            logger.error(f"Plan answer synthesis failed: {e}", exc_info=True)
            return str(result.result)

    async def _extract_plan_slots(
        self, slot_names: list[str], user_input: str
    ) -> dict[str, str] | None:
        """Extract a cached plan's slot values from a new phrasing.

        All-or-nothing: if any required slot is absent we return None and
        the caller falls through to fresh planning (which may legitimately
        produce a different plan for the variant phrasing).
        """
        if not slot_names:
            return {}
        try:
            model = self.get_model(ModelTier.LIGHT)
        except ValueError:
            return None
        prompt = (
            "Extract these slot values from the user input. A value must be "
            "the SHORTEST literal span copied verbatim from the input. If "
            "any slot's value is absent, respond with exactly: slots={}\n"
            f"Slots: {', '.join(slot_names)}\n"
            f"User input: {user_input}\n"
            'Respond with ONLY one line: slots={"name": "value"}'
        )
        try:
            response = await model.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=120,
            )
        except Exception as e:
            logger.error(f"Plan slot extraction failed: {e}", exc_info=True)
            return None
        slots = IntentRouter._parse_slots(response.content)
        extracted = {k: v for k, v in slots.items() if k in slot_names and v}
        if set(slot_names) <= set(extracted):
            return extracted
        return None

    def _after_plan_success(
        self,
        plan_id: str,
        source_query: str = "",
        slots: dict[str, str] | None = None,
    ) -> None:
        """Bookkeeping after a plan run succeeds; rule-track auto-promotion."""
        promotable = self.plan_store.record_success(plan_id, source_query, slots)
        if promotable and self.config.plan_auto_promote:
            self.promote_plan(plan_id)

    def approve_plan(self, plan_id: str) -> bool:
        """Open the reuse gate for a candidate plan (manual track of
        ``plan_auto_reuse``): from now on new requests may match and reuse
        it. Promotion to a Scenario stays a separate, later decision."""
        return self.plan_store.approve(plan_id)

    def promote_plan(self, plan_id: str) -> bool:
        """Promote a cached plan into a registered Scenario (manual track,
        also used by rule-track auto-promotion).

        From this point the classifier routes matching requests through the
        Scenario path — the plan's recorded user phrasings serve as the
        retrieval prefilter's trigger anchors.
        """
        scenario = self.plan_store.to_scenario_config(plan_id)
        if scenario is None:
            return False
        self.register_scenario(plan_id, scenario)
        self.plan_store.mark_promoted(plan_id)
        logger.info(f"Plan promoted to scenario: {plan_id}")
        return True

    def list_plan_candidates(self) -> list[dict[str, Any]]:
        """Cached plans with their stats — for review before manual promotion."""
        return [plan.model_dump() for plan in self.plan_store.list_plans()]

    def export_plan_scenario(self, plan_id: str) -> ScenarioConfig | None:
        """Render a cached plan as a ScenarioConfig draft the developer can
        codify (the durable, human-confirmed promotion track)."""
        return self.plan_store.to_scenario_config(plan_id)

    def _record_react_transcript(
        self, context: SessionContext, actions: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Log a completed ReAct run's tool chain for D5 transcript mining.

        No-op unless ``enable_transcript_mining`` is on, or the run didn't
        chain at least 2 successful tool calls (nothing to mine). Bounded
        FIFO log; ``mine_scenario_candidates()`` drains it.
        """
        if not self.config.enable_transcript_mining or len(actions) < 2:
            return
        self._react_transcripts.append(
            ReactTranscript(user_input=context.user_input, actions=list(actions))
        )
        overflow = len(self._react_transcripts) - self.config.mining_max_transcripts
        if overflow > 0:
            del self._react_transcripts[:overflow]

    def mine_scenario_candidates(self) -> list[str]:
        """Cluster logged ReAct transcripts into plan_store candidates (D5).

        Call this periodically from an operator-driven background task —
        the framework has no built-in scheduler. Drains the transcript log
        on every call: a tool-chain shape that hasn't yet recurred
        ``mining_min_cluster_size`` times within one batch is dropped with
        it rather than held over, mirroring how the Planner path treats a
        one-off request — not enough evidence yet, cheap to observe again
        on the next batch. Mined candidates land in the same reviewable
        ``plan_store`` a planner-generated candidate would, unapproved and
        unpromoted until a developer (or the existing rule tracks) acts.
        """
        touched = self.transcript_miner.mine(self._react_transcripts, self.plan_store)
        self._react_transcripts.clear()
        return touched

    async def _direct_response(
        self, context: SessionContext, adapter: SSEStreamAdapter | None = None
    ) -> str:
        """Generate a direct LLM response (no tools needed)."""
        model = self.get_model(ModelTier.HEAVY)

        system_prompt = self.prompt_manager.get_output_prompt()
        if self.config.system_prompt:
            system_prompt = self.config.system_prompt

        messages = [{"role": "system", "content": system_prompt}]
        # Inject layered memory: L4 summary + L3 reference as a system message,
        # then L2 verbatim turns as alternating user/assistant messages.
        mem = await self.memory.get(context.session_id, context.user_id)
        messages.extend(mem.to_chat_messages(l2_rounds=5))
        messages.append({"role": "user", "content": context.user_input})

        if adapter:
            # Stream the response
            await adapter.send_event(SSEEventBuilder.answer_start())
            full_response = ""
            async for chunk in model.stream_chat(messages):
                full_response += chunk
                await adapter.send_event(SSEEventBuilder.answer_chunk(chunk))
            return full_response
        else:
            response = await model.chat(messages)
            return response.content

    async def _stream_answer(
        self, answer: str, adapter: SSEStreamAdapter, session_id: str, request_id: str
    ) -> None:
        """Stream a pre-computed answer chunk by chunk."""
        await adapter.send_event(SSEEventBuilder.answer_start())
        # Send the whole answer as one chunk for non-streaming paths
        await adapter.send_event(
            SSEEventBuilder.answer_chunk(answer, conversation_id=session_id, message_id=request_id)
        )
        await adapter.send_event(SSEEventBuilder.answer_end(answer))
