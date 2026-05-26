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
from ..tools.scenario import ScenarioConfig, ScenarioEngine
from ..tools.termination import TerminationChecker
from .cache import CacheManager
from .hooks import HookContext, HookEvent, HookRegistry, HookResult
from .memory_hooks import TopicChangeHook
from .memory_layers import (
    InMemoryBackend,
    LayeredMemoryConfig,
    LayeredMemoryStore,
    _default_summarizer_factory,
)
from .model_client import ModelClient
from .parameter import ParameterManager
from .pipeline import RequestPipeline
from .prompt import PromptManager
from .router import IntentLevel, IntentResult, IntentRouter

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
        self.router = IntentRouter()
        self.pipeline = RequestPipeline()
        self.termination_checker = TerminationChecker()
        self.hooks = HookRegistry()

        # Knowledge base
        self._knowledge_base: KnowledgeBase | None = None

        # Middleware
        self._middleware_chain = MiddlewareChain()

        # Max iterations
        self.max_iterations = self.config.max_iterations

        # Built-in hooks (opt-out via config flags).
        if self.config.memory_enable_topic_change_hook:
            self.hooks.register(TopicChangeHook())

        # MCP clients spun up via register_mcp_server() (kept alive here so
        # they aren't GC'd until shutdown_mcp_servers()).
        self._mcp_clients: dict[str, list[Any]] = {}

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
        self.router.register_scenarios({scenario_id: {"name": scenario.name, "description": scenario.description}})

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

    async def run(self, user_input: str, **context_vars: Any) -> AgentResponse:
        """
        Process a user request (non-streaming).

        Args:
            user_input: User's input text
            **context_vars: Additional context (user_id, session_id, etc.)

        Returns:
            AgentResponse with the final answer
        """
        self._validate_input(user_input)
        request_id = str(uuid.uuid4())
        start_time = time.time()

        session_id = context_vars.get("session_id", f"session_{uuid.uuid4().hex[:8]}")
        user_id = context_vars.get("user_id", "anonymous")

        context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            max_iterations=self.max_iterations,
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
                    await mem.add_turn(user_input, answer)
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
                    await mem.add_turn(user_input, answer)
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
                await mem.add_turn(user_input, answer)
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

            # Execute based on intent
            if intent.level == IntentLevel.SCENARIO and intent.scenario:
                answer = await self._execute_scenario(intent.scenario, context)
            elif intent.level == IntentLevel.REACT:
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
            await mem.add_turn(user_input, answer)

            elapsed_ms = (time.time() - start_time) * 1000
            response = AgentResponse(
                session_id=session_id,
                request_id=request_id,
                answer=answer,
                total_iterations=context.current_iteration,
                execution_time_ms=elapsed_ms,
                metadata={"intent_level": intent.level.value, "scenario": intent.scenario},
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
            return AgentResponse(
                session_id=session_id,
                request_id=request_id,
                answer=self._sanitize_error(e),
                total_iterations=context.current_iteration,
                execution_time_ms=elapsed_ms,
                metadata={"error": str(e)} if self.config.debug else {},
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
        self._validate_input(request.user_input)
        request_id = str(uuid.uuid4())
        start_time = time.time()

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
                    await mem.add_turn(request.user_input, answer)
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

            # Execute
            if intent.level == IntentLevel.SCENARIO and intent.scenario:
                answer = await self._execute_scenario(intent.scenario, context)
            elif intent.level == IntentLevel.REACT:
                answer = await self._react_loop(context, adapter)
            else:
                answer = await self._direct_response(context, adapter)

            answer = await self.on_before_respond(context, answer)
            await self._dispatch_hook(
                HookEvent.BEFORE_RESPOND,
                user_input=request.user_input, session_id=request.session_id,
                user_id=request.user_id, memory=mem, answer=answer,
            )
            await self._stream_answer(answer, adapter, request.session_id, request_id)
            await adapter.send_event(SSEEventBuilder.completed())
            await adapter.finish()

            await mem.add_turn(request.user_input, answer)
            elapsed_ms = (time.time() - start_time) * 1000

            response = AgentResponse(
                session_id=request.session_id, request_id=request_id,
                answer=answer, total_iterations=context.current_iteration,
                execution_time_ms=elapsed_ms,
                metadata={"intent_level": intent.level.value, "scenario": intent.scenario},
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
            await adapter.send_event(SSEEventBuilder.error(error_msg))
            await adapter.finish()
            return AgentResponse(
                session_id=request.session_id, request_id=request_id,
                answer=error_msg, total_iterations=context.current_iteration,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)} if self.config.debug else {},
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

        for iteration in range(self.max_iterations):
            context.current_iteration = iteration + 1

            if adapter:
                await adapter.send_event(SSEEventBuilder.thought_start(iteration + 1))

            # Generate thought
            thought = await self._generate_thought(context, model, accumulated_context)

            if adapter:
                await adapter.send_event(SSEEventBuilder.thought_end(thought, iteration + 1))

            context.add_step("THOUGHT", thought)

            # Check termination
            should_stop, reason = self.termination_checker.should_terminate(context, thought)
            if should_stop:
                logger.info(f"ReAct loop terminated: {reason}")
                break

            # Parse action from thought
            tool_name, tool_params = self._parse_action(thought)

            if tool_name:
                if adapter:
                    await adapter.send_event(SSEEventBuilder.action_start(tool_name, iteration + 1))

                await self.on_before_tool_call(context, tool_name, tool_params)
                result = await self.tool_executor.execute(tool_name, context, **tool_params)
                await self.on_after_tool_call(context, tool_name, result)

                observation = str(result.result) if result.success else f"Error: {result.error}"
                accumulated_context += f"\nTool: {tool_name}\nResult: {observation}\n"

                context.add_step("ACTION", f"{tool_name}({tool_params})")
                context.add_step("OBSERVATION", observation)

                if adapter:
                    await adapter.send_event(
                        SSEEventBuilder.action_end(tool_name, tool_params, iteration + 1)
                    )
                    await adapter.send_event(SSEEventBuilder.observation(observation, iteration + 1))

                # Check if the tool returned a direct output
                if result.success and result.output_type == ToolOutputType.DIRECT_OUTPUT:
                    return observation
            else:
                # No action — extract answer from thought
                answer = self._extract_final_answer(thought)
                if answer:
                    return answer

        # Generate final answer from accumulated context
        return await self._generate_final_answer(context, model, accumulated_context)

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

    async def _execute_scenario(self, scenario_id: str, context: SessionContext) -> str:
        """Execute a pre-defined scenario toolchain."""
        result = await self.scenario_engine.execute(
            scenario_id, context, self.tool_executor,
            extra_vars={
                "user_id": context.user_id,
                "user_input": context.user_input,
                "session_id": context.session_id,
            },
        )

        if result.success:
            if result.output_type == ToolOutputType.DIRECT_OUTPUT:
                return str(result.result)
            else:
                # Use heavy model to format the result
                model = self.get_model(ModelTier.HEAVY)
                prompt = (
                    f"{self.prompt_manager.get_output_prompt()}\n\n"
                    f"Tool results: {result.result}\n\n"
                    f"User question: {context.user_input}\n\n"
                    "Generate a natural, helpful response."
                )
                response = await model.chat([{"role": "user", "content": prompt}])
                return response.content
        else:
            return f"Sorry, I couldn't complete this request: {result.error}"

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
