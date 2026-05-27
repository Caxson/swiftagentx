"""
OpenAI-compatible model client.

Supports any service that implements the OpenAI chat completions API,
including OpenAI, Azure OpenAI, DeepSeek, DashScope, Together AI,
and other OpenAI-compatible endpoints.
"""

import asyncio
import json
import logging
import queue
import threading
from collections.abc import AsyncGenerator
from typing import Any

from ..core.model_client import ModelClient, ModelResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(ModelClient):
    """
    OpenAI-compatible model client using httpx or requests.

    Args:
        api_key: API key
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
        api_base: API base URL (e.g., "https://api.openai.com/v1")
        temperature: Temperature parameter
        max_tokens: Maximum tokens
        timeout_seconds: Request timeout
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout_seconds: int = 60,
    ):
        # Fail fast at construction time with a clear actionable message
        # instead of crashing inside the first chat() call with the wrong
        # error name. ``swiftagentx[openai]`` brings httpx with the right
        # extras.
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleProvider requires the 'httpx' package. "
                "Install it with: pip install 'swiftagentx[openai]'"
            ) from exc
        super().__init__(
            api_key=api_key, model=model, temperature=temperature,
            max_tokens=max_tokens, timeout_seconds=timeout_seconds,
        )
        self.api_base = api_base.rstrip("/")

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "presence_penalty": kwargs.get("presence_penalty", 0),
        }

    async def complete(self, prompt: str, **kwargs: Any) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        url = f"{self.api_base}/chat/completions"
        payload = self._build_payload(messages, **kwargs)
        headers = self._get_headers()

        try:
            # Try httpx first, fall back to requests
            try:
                import httpx
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
            except ImportError:
                import requests as req
                response_sync = await asyncio.to_thread(
                    lambda: req.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
                )
                if response_sync.status_code != 200:
                    raise Exception(f"API error: {response_sync.status_code} - {response_sync.text}")
                data = response_sync.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            return ModelResponse(
                content=content,
                model=self.model,
                tokens_used=usage.get("total_tokens", 0),
                metadata={
                    "finish_reason": data["choices"][0].get("finish_reason", ""),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            )
        except Exception as e:
            logger.error(f"Chat request failed: {e}", exc_info=True)
            raise

    async def stream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.stream_chat(messages, **kwargs):
            yield chunk

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        url = f"{self.api_base}/chat/completions"
        payload = self._build_payload(messages, stream=True, **kwargs)
        headers = self._get_headers()

        # Try httpx async streaming first
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
            return
        except ImportError:
            pass

        # Fallback: requests + background thread
        import requests as req

        result_queue: queue.Queue = queue.Queue()

        def _stream_in_thread() -> None:
            try:
                resp = req.post(url, headers=headers, json=payload, timeout=self.timeout_seconds, stream=True)
                if resp.status_code != 200:
                    result_queue.put(("ERROR", f"API error: {resp.status_code} - {resp.text}"))
                    return
                for line in resp.iter_lines(decode_unicode=False):
                    if not line:
                        continue
                    try:
                        line_str = line.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        continue
                    if not line_str.startswith("data: "):
                        continue
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        result_queue.put(("DONE", None))
                        return
                    try:
                        data = json.loads(data_str)
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            result_queue.put(("CONTENT", content))
                    except json.JSONDecodeError:
                        continue
                result_queue.put(("DONE", None))
            except Exception as e:
                result_queue.put(("ERROR", str(e)))

        thread = threading.Thread(target=_stream_in_thread, daemon=True)
        thread.start()

        while True:
            try:
                item = await asyncio.to_thread(result_queue.get, timeout=0.1)
                msg_type, content = item
                if msg_type == "CONTENT":
                    yield content
                elif msg_type == "DONE":
                    break
                elif msg_type == "ERROR":
                    raise Exception(content)
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

        thread.join(timeout=1.0)
