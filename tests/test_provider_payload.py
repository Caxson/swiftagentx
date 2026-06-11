"""Tests for OpenAICompatibleProvider payload construction (extra_params passthrough).

Vendor-specific request fields (DashScope's ``enable_thinking``, vLLM's
``chat_template_kwargs``, ...) must reach the wire without the framework
hardcoding every vendor's knobs.
"""

from swiftagentx.providers.openai_compatible import OpenAICompatibleProvider


def _provider(**kwargs) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(api_key="k", model="m", **kwargs)


class TestExtraParams:
    def test_constructor_extra_params_land_in_payload(self):
        p = _provider(extra_params={"enable_thinking": False})
        payload = p._build_payload([{"role": "user", "content": "hi"}])
        assert payload["enable_thinking"] is False

    def test_per_call_extra_params_override_constructor(self):
        p = _provider(extra_params={"enable_thinking": False})
        payload = p._build_payload(
            [{"role": "user", "content": "hi"}],
            extra_params={"enable_thinking": True, "seed": 7},
        )
        assert payload["enable_thinking"] is True
        assert payload["seed"] == 7

    def test_extra_params_cannot_clobber_messages_or_stream(self):
        p = _provider(extra_params={"messages": [], "stream": True, "seed": 1})
        messages = [{"role": "user", "content": "hi"}]
        payload = p._build_payload(messages, stream=False)
        assert payload["messages"] == messages
        assert payload["stream"] is False
        assert payload["seed"] == 1

    def test_no_extra_params_payload_unchanged(self):
        payload = _provider()._build_payload(
            [{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=100,
        )
        assert set(payload) == {
            "model", "messages", "temperature", "max_tokens", "stream",
            "top_p", "frequency_penalty", "presence_penalty",
        }
        assert payload["temperature"] == 0.1
