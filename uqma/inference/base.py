"""Backend protocol.

Stage 1 of the pipeline logs raw signal and commits to nothing (plan §6.2), so a backend
returns the generation plus whatever per-token evidence it can, and never a score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Generation:
    text: str
    token_logprobs: list[float] | None = None
    n_generated_tokens: int | None = None
    finish_reason: str | None = None
    truncated_by_context: bool = False
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    """Anything that can continue a chat transcript.

    ``generate`` takes the full message list rather than a rendered string so that a
    backend owns its own chat templating -- and so that k-resampling at turn t with
    history pinned (plan §6.4) is just calling it k times with the same list.
    """

    def generate(
        self, messages: list[dict], capture_logprobs: bool = True, **kwargs
    ) -> Generation: ...
