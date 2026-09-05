"""Backends that need no GPU, for tests and the smoke pipeline.

The smoke pipeline (plan §6.4) runs all stages on three tasks in under two minutes with
no GPU and no Docker. It exists to catch a config error before it eats a 15-GPU-hour
sweep, so it must never require the real stack.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .base import Generation


class ScriptedBackend:
    """Replays a fixed list of responses, optionally per task.

    ``responses`` may be a flat sequence (consumed in order across the whole run) or a
    mapping from task id to that task's sequence. The task id is recovered from the
    opening user message, which is the only place it appears in a transcript.
    """

    def __init__(
        self,
        responses: Sequence[str] | dict[str, Sequence[str]],
        default: str = "FINISH([])",
    ) -> None:
        self._responses = responses
        self._default = default
        self._cursor = 0
        self._per_task: dict[str, int] = {}
        self.calls: list[list[dict]] = []

    def generate(
        self, messages: list[dict], capture_logprobs: bool = True, **kwargs
    ) -> Generation:
        self.calls.append(list(messages))

        if isinstance(self._responses, dict):
            key = _task_key(messages, self._responses)
            seq = self._responses.get(key, ())
            index = self._per_task.get(key, 0)
            self._per_task[key] = index + 1
            text = seq[index] if index < len(seq) else self._default
        else:
            text = (
                self._responses[self._cursor]
                if self._cursor < len(self._responses)
                else self._default
            )
            self._cursor += 1

        n_tokens = max(len(text.split()), 1)
        return Generation(
            text=text,
            token_logprobs=[-0.1] * n_tokens if capture_logprobs else None,
            n_generated_tokens=n_tokens,
            finish_reason="stop",
        )


class CallableBackend:
    """Wraps a function of the transcript, for tests that need branching behaviour."""

    def __init__(self, fn: Callable[[list[dict]], str]) -> None:
        self._fn = fn
        self.calls: list[list[dict]] = []

    def generate(
        self, messages: list[dict], capture_logprobs: bool = True, **kwargs
    ) -> Generation:
        self.calls.append(list(messages))
        text = self._fn(messages)
        n_tokens = max(len(text.split()), 1)
        return Generation(
            text=text,
            token_logprobs=[-0.1] * n_tokens if capture_logprobs else None,
            n_generated_tokens=n_tokens,
            finish_reason="stop",
        )


def _task_key(messages: list[dict], table: dict[str, Sequence[str]]) -> str:
    opening = messages[0]["content"] if messages else ""
    for key in table:
        if key in opening:
            return key
    return ""
