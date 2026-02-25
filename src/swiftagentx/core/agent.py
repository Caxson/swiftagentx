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

import asyncio
import re
import time
import uuid
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models.schema import (
    AgentRequest, AgentResponse, SessionContext, ContextParameters,
)
from ..models.config import SwiftAgentConfig, ModelTier
from .memory import SessionMemory
from .model_client import ModelClient, ModelResponse, DummyModelClient
from .cache import CacheManager
from .prompt import PromptManager
from .parameter import ParameterManager
from .router import IntentLevel, IntentResult, IntentRouter
from .pipeline import RequestPipeline, StageResult
from ..tools.base import Tool, ToolOutput, ToolOutputType
from ..tools.registry import ToolRegistry
from ..tools.executor import ToolExecutor
from ..tools.termination import TerminationChecker
from ..tools.scenario import ScenarioConfig, ScenarioEngine
from ..stream.adapter import SSEStreamAdapter
from ..stream.builder import SSEEventBuilder
from ..middleware.base import Middleware, MiddlewareChain
from ..knowledge_base.base import KnowledgeBase
from ..knowledge_base.tool import KnowledgeBaseTool

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
        models: Optional[Dict[ModelTier, ModelClient]] = None,
        model: Optional[ModelClient] = None,
        max_iterations: int = 10,
        config: Optional[SwiftAgentConfig] = None,
    ):
        self.name = name
        self.config = config or SwiftAgentConfig(name=name, max_iterations=max_iterations)

        # Model setup
        self._models: Dict[ModelTier, ModelClient] = models or {}
        if model is not None:
            if ModelTier.HEAVY not in self._models:
                self._models[ModelTier.HEAVY] = model
            if ModelTier.LIGHT not in self._models:
                self._models[ModelTier.LIGHT] = model

        # Subsystems
        self.memory = SessionMemory()
        self.cache = CacheManager()
        self.prompt_manager = PromptManager()
        self.parameter_manager = ParameterManager()
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.scenario_engine = ScenarioEngine()
        self.router = IntentRouter()
        self.pipeline = RequestPipeline()
        self.termination_checker = TerminationChecker()

        # Knowledge base
        self._knowledge_base: Optional[KnowledgeBase] = None

        # Middleware
        self._middleware_chain = MiddlewareChain()

        # Max iterations
        self.max_iterations = self.config.max_iterations

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

    def set_knowledge_base(self, kb: KnowledgeBase) -> None:
        """
        Attach a knowledge base to this agent.

        Automatically registers a KnowledgeBaseTool so the agent
        can query the KB during ReAct loops.
        """
        self._knowledge_base = kb
        kb_tool = KnowledgeBaseTool(
            kb=kb,
            exact_match_threshold=self.config.kb_exact_match_threshold,
        )
        self.tool_registry.register(kb_tool)

    @property
    def knowledge_base(self) -> Optional[KnowledgeBase]:
        return self._knowledge_base

    def register_tool(self, tool: Tool) -> None:
        self.tool_registry.register(tool)

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

        try:
            # Lifecycle hook
            await self.on_request_start(context)

            # Add user message to memory
            self.memory.add_message("user", user_input)

            # Cache check
            if self.config.enable_cache:
                hit, cached_value, source = self.cache.query(
                    user_input, user_id, session_id,
                    platform=context_vars.get("platform", "default"),
                )
                if hit:
                    answer = str(cached_value)
                    self.memory.add_message("assistant", answer)
                    return AgentResponse(
                        session_id=session_id,
                        request_id=request_id,
                        answer=answer,
                        total_iterations=0,
                        execution_time_ms=(time.time() - start_time) * 1000,
                        metadata={"cache_hit": True, "cache_source": source},
                    )

            # Intent classification
            await self.on_before_classify(context)
            intent = await self._classify_intent(user_input, context)
            await self.on_after_classify(context, intent)

            # Execute based on intent
            if intent.level == IntentLevel.SCENARIO and intent.scenario:
                answer = await self._execute_scenario(intent.scenario, context)
            elif intent.level == IntentLevel.REACT:
                answer = await self._react_loop(context)
            else:
                answer = await self._direct_response(context)

            # Post-processing hook
            answer = await self.on_before_respond(context, answer)

            # Add to memory
            self.memory.add_message("assistant", answer)

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

        try:
            await self.on_request_start(context)
            await adapter.send_event(SSEEventBuilder.initialized(f"{self.name} initialized"))
            self.memory.add_message("user", request.user_input)

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
                    self.memory.add_message("assistant", answer)
                    return AgentResponse(
                        session_id=request.session_id, request_id=request_id,
                        answer=answer, total_iterations=0,
                        execution_time_ms=(time.time() - start_time) * 1000,
                        metadata={"cache_hit": True, "cache_source": source},
                    )

            # Classification
            await self.on_before_classify(context)
            intent = await self._classify_intent(request.user_input, context)
            await self.on_after_classify(context, intent)

            # Execute
            if intent.level == IntentLevel.SCENARIO and intent.scenario:
                answer = await self._execute_scenario(intent.scenario, context)
            elif intent.level == IntentLevel.REACT:
                answer = await self._react_loop(context, adapter)
            else:
                answer = await self._direct_response(context, adapter)

            answer = await self.on_before_respond(context, answer)
            await self._stream_answer(answer, adapter, request.session_id, request_id)
            await adapter.send_event(SSEEventBuilder.completed())
            await adapter.finish()

            self.memory.add_message("assistant", answer)
            elapsed_ms = (time.time() - start_time) * 1000

            response = AgentResponse(
                session_id=request.session_id, request_id=request_id,
                answer=answer, total_iterations=context.current_iteration,
                execution_time_ms=elapsed_ms,
                metadata={"intent_level": intent.level.value, "scenario": intent.scenario},
            )
            await self.on_request_end(context, response)
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

    async def on_before_tool_call(self, context: SessionContext, tool_name: str, params: Dict[str, Any]) -> None:
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
        self, context: SessionContext, adapter: Optional[SSEStreamAdapter] = None
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

    def _parse_action(self, thought: str) -> Tuple[str, Dict[str, Any]]:
        """Parse tool name and parameters from a thought."""
        import json as json_mod

        # Match "Action: tool_name"
        action_match = re.search(r"Action:\s*(\w+)", thought)
        if not action_match:
            return "", {}

        tool_name = action_match.group(1)

        # Match "Action Input: {...}"
        input_match = re.search(r"Action Input:\s*(\{.*?\})", thought, re.DOTALL)
        params: Dict[str, Any] = {}
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
        self, context: SessionContext, adapter: Optional[SSEStreamAdapter] = None
    ) -> str:
        """Generate a direct LLM response (no tools needed)."""
        model = self.get_model(ModelTier.HEAVY)

        system_prompt = self.prompt_manager.get_output_prompt()
        if self.config.system_prompt:
            system_prompt = self.config.system_prompt

        messages = [{"role": "system", "content": system_prompt}]
        # Add conversation history
        history = self.memory.get_conversation_for_reply(rounds=5)
        messages.extend(history)
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
