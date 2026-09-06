"""k-resampling at a turn with history held fixed.

The sampling protocol resamples the *action* at turn t rather than whole trajectories,
which keeps cost at k x instead of combinatorial (proposal §6.3). Two properties matter
downstream:

* The k samples share a long identical prompt, so vLLM must be served with
  ``--enable-prefix-caching`` or the prefill is recomputed k times.
* Samples are recorded, never executed. The episode continues along the greedy action,
  so sampling does not multiply environment interactions -- and in this environment it
  could not corrupt state anyway, since POSTs are never sent.

Estimators are computed in stage 4 off the logged samples (plan §6.2); this module only
collects evidence.
"""

from __future__ import annotations

from collections import Counter

from ..inference.base import Backend
from ..trajectory.schema import Sample
from .parsing import canonical_action, parse


def resample_turn(
    messages: list[dict],
    backend: Backend,
    k: int,
    temperature: float = 1.0,
    capture_logprobs: bool = True,
    strip_think: bool = False,
) -> list[Sample]:
    """Draw k alternative actions for the same history.

    ``messages`` must be the history *before* the turn's own generation is appended, so
    that every sample sees exactly what the greedy action saw.
    """
    samples: list[Sample] = []
    for index in range(k):
        generation = backend.generate(
            messages, capture_logprobs=capture_logprobs, temperature=temperature
        )
        action = parse(generation.text, strip_think=strip_think)
        samples.append(
            Sample(
                index=index,
                text=generation.text,
                action_kind=action.kind.value,
                action_raw=action.raw,
                token_logprobs=generation.token_logprobs,
                n_generated_tokens=generation.n_generated_tokens,
                finish_reason=generation.finish_reason,
                canonical_action=canonical_action(action),
            )
        )
    return samples


def action_distribution(samples: list[Sample]) -> Counter[str]:
    """Counts over canonical actions.

    Exact-match equivalence is sound here because actions are structured tool calls: two
    samples mean the same thing iff they are the same call with the same arguments. This
    is what removes the natural-language-inference model that semantic entropy normally
    needs, and with it the largest source of noise in the sampling family.
    """
    return Counter(s.canonical_action or "" for s in samples)


def self_consistency(samples: list[Sample]) -> float | None:
    """Share of samples agreeing with the modal action, in [0, 1].

    Reported as a convenience for smoke-testing the pipeline. The estimators proper live
    in stage 4 and are fitted and evaluated there, not here.
    """
    if not samples:
        return None
    counts = action_distribution(samples)
    return counts.most_common(1)[0][1] / len(samples)


def distinct_actions(samples: list[Sample]) -> int:
    return len(action_distribution(samples))
