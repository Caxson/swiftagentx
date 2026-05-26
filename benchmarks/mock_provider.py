"""
Mock LLM provider for deterministic benchmarking.

Simulates realistic latency distributions for "light" and "heavy" model
tiers, plus a per-call token cost. No API key, no network, no flakiness —
the same seed produces the same measurements every run.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from swiftagentx.core.model_client import ModelClient, ModelResponse


@dataclass
class LatencyProfile:
    """Truncated log-normal latency profile in milliseconds."""

    mean_ms: float
    stddev_ms: float
    floor_ms: float = 10.0

    def sample(self, rng: random.Random) -> float:
        ms = max(self.floor_ms, rng.gauss(self.mean_ms, self.stddev_ms))
        return ms


# Calibrated against public benchmarks (gpt-4o-mini ~ 300ms TTFT,
# gpt-4-class ~ 1500ms TTFT). These are intentionally rough but realistic.
LIGHT_PROFILE = LatencyProfile(mean_ms=180.0, stddev_ms=50.0, floor_ms=80.0)
HEAVY_PROFILE = LatencyProfile(mean_ms=1400.0, stddev_ms=400.0, floor_ms=600.0)


@dataclass
class MockCallStats:
    """Tracks calls and tokens across one benchmark run."""

    light_calls: int = 0
    heavy_calls: int = 0
    light_tokens: int = 0
    heavy_tokens: int = 0
    responses: list[str] = field(default_factory=list)

    @property
    def total_calls(self) -> int:
        return self.light_calls + self.heavy_calls

    @property
    def total_tokens(self) -> int:
        return self.light_tokens + self.heavy_tokens

    def reset(self) -> None:
        self.light_calls = 0
        self.heavy_calls = 0
        self.light_tokens = 0
        self.heavy_tokens = 0
        self.responses = []


class MockModelClient(ModelClient):
    """
    Deterministic mock LLM with configurable response templates.

    Use ``set_response_map`` to control what the model "decides" for each
    classified intent — this lets benchmark scenarios drive the agent through
    specific execution paths (cache hit, scenario, ReAct, direct).
    """

    def __init__(
        self,
        api_key: str = "mock",
        model: str = "mock-model",
        tier: str = "light",
        seed: int = 42,
        stats: MockCallStats | None = None,
    ) -> None:
        super().__init__(api_key=api_key, model=model)
        self.tier = tier
        self.profile = LIGHT_PROFILE if tier == "light" else HEAVY_PROFILE
        self.rng = random.Random(seed)
        self.stats = stats or MockCallStats()
        self._response_map: dict[str, str] = {}
        self._default_response = "This is a mocked response."

    def set_response_map(self, mapping: dict[str, str]) -> None:
        """Map a keyword (substring of prompt) to a canned response."""
        self._response_map = mapping

    def set_default_response(self, response: str) -> None:
        self._default_response = response

    def _pick_response(self, prompt: str) -> str:
        for keyword, response in self._response_map.items():
            if keyword.lower() in prompt.lower():
                return response
        return self._default_response

    async def _simulate_latency(self) -> None:
        ms = self.profile.sample(self.rng)
        await asyncio.sleep(ms / 1000.0)

    def _account_call(self, prompt: str, response: str) -> None:
        tokens = max(1, (len(prompt) + len(response)) // 4)
        if self.tier == "light":
            self.stats.light_calls += 1
            self.stats.light_tokens += tokens
        else:
            self.stats.heavy_calls += 1
            self.stats.heavy_tokens += tokens
        self.stats.responses.append(response)

    async def complete(self, prompt: str, **kwargs: Any) -> ModelResponse:
        await self._simulate_latency()
        response = self._pick_response(prompt)
        self._account_call(prompt, response)
        return ModelResponse(content=response, model=self.model, tokens_used=len(response) // 4)

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        await self._simulate_latency()
        prompt = " ".join(m.get("content", "") for m in messages)
        response = self._pick_response(prompt)
        self._account_call(prompt, response)
        return ModelResponse(content=response, model=self.model, tokens_used=len(response) // 4)

    async def stream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        response = (await self.complete(prompt, **kwargs)).content
        for ch in response:
            yield ch

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        response = (await self.chat(messages, **kwargs)).content
        for ch in response:
            yield ch
