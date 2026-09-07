"""OpenAI-compatible backend against a faked HTTP session.

The real server is `vllm serve`, which is not runnable here, so the contract is pinned
against recorded response shapes instead. Logprob extraction matters most: logprobs are
the measurement instrument for the whole thesis, and a silent None would look like a
model that emits nothing rather than a server that was never asked.
"""

from __future__ import annotations

import json

import pytest

from uqma.inference.openai_compat import OpenAICompatBackend


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or json.dumps(payload)
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, response):
        self._response = response
        self.headers = {}
        self.posts: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        return self._response

    def get(self, url, timeout=None):
        return FakeResponse({"data": [{"id": "fake-model"}]})


def _completion(text, logprobs=None, finish_reason="stop"):
    choice = {"message": {"content": text}, "finish_reason": finish_reason}
    if logprobs is not None:
        choice["logprobs"] = {"content": [{"token": "x", "logprob": lp} for lp in logprobs]}
    return {"choices": [choice], "usage": {"completion_tokens": 7, "prompt_tokens": 1200}}


def test_generation_and_logprob_extraction():
    session = FakeSession(FakeResponse(_completion("FINISH([1.2])", [-0.1, -0.5, -0.02])))
    backend = OpenAICompatBackend(model="m", session=session)
    gen = backend.generate([{"role": "user", "content": "hi"}])

    assert gen.text == "FINISH([1.2])"
    assert gen.token_logprobs == [-0.1, -0.5, -0.02]
    assert gen.n_generated_tokens == 7
    assert gen.meta["prompt_tokens"] == 1200
    assert not gen.truncated_by_context


def test_logprobs_requested_only_when_asked():
    session = FakeSession(FakeResponse(_completion("x")))
    backend = OpenAICompatBackend(model="m", session=session)

    backend.generate([{"role": "user", "content": "hi"}], capture_logprobs=True)
    assert session.posts[-1]["json"]["logprobs"] is True

    backend.generate([{"role": "user", "content": "hi"}], capture_logprobs=False)
    assert "logprobs" not in session.posts[-1]["json"]


def test_absent_logprobs_are_none_not_empty():
    session = FakeSession(FakeResponse(_completion("x", logprobs=None)))
    backend = OpenAICompatBackend(model="m", session=session)
    assert backend.generate([{"role": "user", "content": "hi"}]).token_logprobs is None


def test_context_overflow_is_flagged_not_raised():
    session = FakeSession(
        FakeResponse({}, status_code=400,
                     text="This model's maximum context length is 8192 tokens")
    )
    backend = OpenAICompatBackend(model="m", session=session)
    gen = backend.generate([{"role": "user", "content": "hi"}])
    assert gen.truncated_by_context
    assert gen.text == ""


def test_other_http_errors_still_raise():
    session = FakeSession(FakeResponse({}, status_code=500, text="internal error"))
    backend = OpenAICompatBackend(model="m", session=session)
    with pytest.raises(RuntimeError):
        backend.generate([{"role": "user", "content": "hi"}])


def test_temperature_defaults_to_zero_for_parity():
    # Gate G4 compares per-task outcomes against upstream at temperature 0.
    session = FakeSession(FakeResponse(_completion("x")))
    backend = OpenAICompatBackend(model="m", session=session)
    backend.generate([{"role": "user", "content": "hi"}])
    assert session.posts[-1]["json"]["temperature"] == 0.0


def test_model_discovery_when_unset():
    session = FakeSession(FakeResponse(_completion("x")))
    backend = OpenAICompatBackend(session=session)
    assert backend.discover_model() == "fake-model"


def test_chat_template_kwargs_are_forwarded():
    """Disabling reasoning has to reach the server, not just our parser.

    Qwen3 emits <think> by default; on MedAgentBench that produced 100% invalid actions,
    and nearly half the turns hit max_tokens mid-thought so stripping could not recover
    them either. The fix has to be at generation time.
    """
    session = FakeSession(FakeResponse(_completion("GET http://x?a=1")))
    backend = OpenAICompatBackend(
        model="m", session=session,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    backend.generate([{"role": "user", "content": "hi"}])
    sent = session.posts[-1]["json"]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_no_extra_body_leaves_the_request_clean():
    session = FakeSession(FakeResponse(_completion("x")))
    OpenAICompatBackend(model="m", session=session).generate([{"role": "user", "content": "hi"}])
    assert "chat_template_kwargs" not in session.posts[-1]["json"]
