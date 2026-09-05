"""Backend speaking the OpenAI-compatible API that ``vllm serve`` exposes.

HTTP rather than the in-process ``vllm.LLM`` class, for three reasons:

* the sweep is data-parallel across independent single-GPU replicas (proposal §8.1), so
  the natural unit is "one server per GPU, many clients";
* this module then imports nothing heavier than ``requests``, so every stage except the
  server itself runs on a laptop;
* logprobs come back in a stable documented shape.

Serve with the flags gate G0 requires::

    vllm serve <MODEL> --dtype bfloat16 --max-model-len 8192 \\
      --enable-prefix-caching --no-enable-chunked-prefill

``--enable-prefix-caching`` matters because k resampled actions at one turn share a long
identical prompt; without it the prefill is recomputed k times. ``--no-enable-chunked-prefill``
is required for hidden-state extraction later (proposal §8.4) and is set now so that the
throughput measured at G2 matches the configuration the real sweeps use.
"""

from __future__ import annotations

import requests

from .base import Generation

# vLLM reports a context overflow as a 400 with this substring; treated as a normal
# episode outcome rather than an error, matching upstream's AGENT_CONTEXT_LIMIT.
_CONTEXT_MARKERS = ("maximum context length", "longer than the maximum", "context_length_exceeded")


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 300.0,
        api_key: str = "EMPTY",
        session: requests.Session | None = None,
        extra_body: dict | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self._session = session or requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def discover_model(self) -> str:
        """Ask the server which model it is serving, so configs need not repeat it."""
        response = self._session.get(f"{self.base_url}/models", timeout=self.timeout)
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            raise RuntimeError(f"no models served at {self.base_url}")
        self.model = data[0]["id"]
        return self.model

    def generate(
        self, messages: list[dict], capture_logprobs: bool = True, **kwargs
    ) -> Generation:
        payload = {
            "model": self.model or self.discover_model(),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            **self.extra_body,
        }
        if capture_logprobs:
            payload["logprobs"] = True

        response = self._session.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout
        )

        if response.status_code == 400 and _is_context_overflow(response.text):
            return Generation(text="", truncated_by_context=True, finish_reason="length")
        response.raise_for_status()
        body = response.json()

        choice = body["choices"][0]
        text = choice["message"]["content"] or ""
        finish_reason = choice.get("finish_reason")
        usage = body.get("usage", {})

        return Generation(
            text=text,
            token_logprobs=_extract_logprobs(choice),
            n_generated_tokens=usage.get("completion_tokens"),
            finish_reason=finish_reason,
            truncated_by_context=finish_reason == "length" and not text.strip(),
            meta={"prompt_tokens": usage.get("prompt_tokens")},
        )


def _is_context_overflow(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _CONTEXT_MARKERS)


def _extract_logprobs(choice: dict) -> list[float] | None:
    """Pull per-token logprobs out of the chat-completions response.

    Shape is ``choices[].logprobs.content[] -> {token, logprob, ...}``. Returns None when
    the server was not asked for logprobs or omitted them, so that a silent absence is
    distinguishable from a genuinely empty generation.
    """
    container = choice.get("logprobs")
    if not container:
        return None
    content = container.get("content")
    if not content:
        return None
    return [item["logprob"] for item in content if "logprob" in item]
