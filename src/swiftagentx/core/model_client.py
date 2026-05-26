"""
Model client — abstract interface for LLM interaction.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)


class ModelResponse:
    """LLM response wrapper."""

    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"<ModelResponse: {self.model}>"


class ModelClient(ABC):
    """
    Base class for all LLM clients.

    Subclass this to integrate any LLM provider.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout_seconds: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Get a completion from the model."""
        ...

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        """Call the model's chat interface."""
        ...

    @abstractmethod
    async def stream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream a completion from the model."""
        ...

    @abstractmethod
    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream a chat response from the model."""
        ...


class DummyModelClient(ModelClient):
    """
    Dummy model client for testing.

    Returns fixed responses without making any API calls.
    """

    async def complete(self, prompt: str, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            content=f"[Dummy response] Completion for: {prompt[:50]}...",
            model=self.model,
            tokens_used=100,
        )

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        last_message = messages[-1]["content"] if messages else ""
        return ModelResponse(
            content=f"[Dummy response] Reply to: {last_message[:50]}...",
            model=self.model,
            tokens_used=100,
        )

    async def stream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        for word in ["This", " is", " a", " dummy", " response"]:
            yield word

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        for word in ["Dummy", " chat", " response"]:
            yield word


class ModelClientFactory:
    """
    Model client factory with tier support (light/heavy).

    Supports registering provider classes, model tiers, and instance caching.
    """

    def __init__(self) -> None:
        self._client_classes: dict[str, type] = {}
        self._model_tiers: dict[str, dict[str, Any]] = {}
        self._client_instances: dict[str, ModelClient] = {}

    def register_client(self, provider: str, client_class: type) -> None:
        self._client_classes[provider] = client_class
        logger.info(f"Registered client class for provider: {provider}")

    def register_model_tier(self, tier: str, provider: str, model_name: str, **config: Any) -> None:
        self._model_tiers[tier] = {"provider": provider, "model": model_name, **config}
        logger.info(f"Registered model tier: {tier} -> {provider}/{model_name}")

    def get_client(
        self,
        provider: str | None = None,
        model_tier: str | None = None,
        **config: Any,
    ) -> ModelClient:
        if model_tier:
            if model_tier not in self._model_tiers:
                raise ValueError(f"Model tier '{model_tier}' not registered")
            tier_config = self._model_tiers[model_tier].copy()
            provider = tier_config.pop("provider")
            model = tier_config.pop("model")
            cache_key = f"{provider}:{model}:{model_tier}"
            if cache_key in self._client_instances:
                return self._client_instances[cache_key]
            if provider not in self._client_classes:
                raise ValueError(f"Provider '{provider}' not registered")
            client = self._client_classes[provider](model=model, **tier_config, **config)
            self._client_instances[cache_key] = client
            return client
        elif provider:
            if provider not in self._client_classes:
                raise ValueError(f"Provider '{provider}' not registered")
            model = config.get("model", "default")
            cache_key = f"{provider}:{model}"
            if cache_key in self._client_instances:
                return self._client_instances[cache_key]
            client = self._client_classes[provider](**config)
            self._client_instances[cache_key] = client
            return client
        else:
            raise ValueError("Either 'provider' or 'model_tier' must be specified")

    def get_light_model(self, **config: Any) -> ModelClient:
        return self.get_client(model_tier="light", **config)

    def get_heavy_model(self, **config: Any) -> ModelClient:
        return self.get_client(model_tier="heavy", **config)

    def list_providers(self) -> list[str]:
        return list(self._client_classes.keys())

    def list_tiers(self) -> list[str]:
        return list(self._model_tiers.keys())

    def clear_cache(self) -> None:
        self._client_instances.clear()

    def clear_all(self) -> None:
        self._client_classes.clear()
        self._model_tiers.clear()
        self._client_instances.clear()
