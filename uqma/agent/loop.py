"""The instrumented ReAct loop.

Replaces upstream's controller/worker/assigner machinery (plan §6.1): we need per-token
logprobs, k-resampling with history pinned, and a teacher-forced second pass, none of
which survive an HTTP boundary between us and the model.

Episode semantics are copied from upstream so gate G4 can compare per-task outcomes:
``max_round`` defaults to 5 (the proposal says 8; Spike A resolves), an unrecognised
action terminates the episode, and a malformed POST body does not.

Instrumentation beyond upstream is additive: turns carry the raw generation, token
logprobs, and optionally k resampled alternatives. Nothing here computes an uncertainty
score -- that is stage 4, off the logged artifact (plan §6.2).
"""

from __future__ import annotations

import time

from ..envs.medagentbench.fhir import FhirClient, PostCheck, check_post_payload
from ..envs.medagentbench.prompts import (
    GET_ERROR,
    GET_OBSERVATION,
    POST_ACCEPTED,
    POST_INVALID,
    build_prompt,
)
from ..envs.medagentbench.tasks import Task
from ..inference.base import Backend, Generation
from ..trajectory.schema import EpisodeStatus, PrefixSource, Trajectory, Turn
from .parsing import Action, ActionKind, canonical_action, parse
from .resample import resample_turn

# Re-exported so callers and tests need not reach into the enum.
STATUS_COMPLETED = EpisodeStatus.COMPLETED.value
STATUS_INVALID_ACTION = EpisodeStatus.AGENT_INVALID_ACTION.value
STATUS_LIMIT_REACHED = EpisodeStatus.TASK_LIMIT_REACHED.value
STATUS_CONTEXT_LIMIT = EpisodeStatus.AGENT_CONTEXT_LIMIT.value
STATUS_ERROR = EpisodeStatus.TASK_ERROR.value


def run_episode(
    task: Task,
    backend: Backend,
    fhir: FhirClient,
    functions: list[dict],
    max_round: int = 5,
    capture_logprobs: bool = True,
    n_samples: int = 0,
    sample_temperature: float = 1.0,
    seed: int | None = None,
    run_id: str | None = None,
) -> Trajectory:
    """Run one task to termination and return the full trajectory record.

    ``n_samples`` > 0 additionally draws k alternative actions at each turn with the
    history held fixed. Those alternatives are recorded but never executed: the episode
    always continues along the greedy action, so sampling costs k x generation and does
    not multiply environment interactions.
    """
    started = time.monotonic()
    opening = build_prompt(task, functions, fhir.api_base)
    history: list[dict] = [{"role": "user", "content": opening}]
    trajectory = Trajectory(
        task_id=task.id,
        category=task.category,
        status=STATUS_LIMIT_REACHED,
        seed=seed,
        run_id=run_id,
    )

    try:
        for turn_index in range(max_round):
            turn_started = time.monotonic()
            generation: Generation = backend.generate(
                history, capture_logprobs=capture_logprobs
            )
            latency = time.monotonic() - turn_started

            if generation.truncated_by_context:
                trajectory.status = STATUS_CONTEXT_LIMIT
                break

            action = parse(generation.text)
            samples = (
                resample_turn(
                    history, backend, k=n_samples, temperature=sample_temperature,
                    capture_logprobs=capture_logprobs,
                )
                if n_samples > 0
                else []
            )

            history.append({"role": "assistant", "content": generation.text})
            observation, post_check = _apply(action, fhir)

            trajectory.turns.append(
                Turn(
                    index=turn_index,
                    generation=generation.text,
                    action_kind=action.kind.value,
                    action_raw=action.raw,
                    observation=observation,
                    url=action.url,
                    post_check=_post_check_dict(post_check),
                    token_logprobs=generation.token_logprobs,
                    n_generated_tokens=generation.n_generated_tokens,
                    finish_reason=generation.finish_reason,
                    samples=samples,
                    prefix_source=PrefixSource.ROLLOUT.value,
                    latency_s=round(latency, 3),
                )
            )

            if action.kind is ActionKind.FINISH:
                trajectory.status = STATUS_COMPLETED
                trajectory.result = action.finish_payload
                break
            if action.kind is ActionKind.INVALID:
                trajectory.status = STATUS_INVALID_ACTION
                break

            history.append({"role": "user", "content": observation})

    except Exception as exc:  # upstream also collapses any error into a failed episode
        trajectory.status = STATUS_ERROR
        trajectory.error = f"{type(exc).__name__}: {exc}"

    trajectory.history = history
    trajectory.wall_s = time.monotonic() - started
    return trajectory


def _apply(action: Action, fhir: FhirClient) -> tuple[str | None, PostCheck | None]:
    """Execute an action against the environment and return the observation text."""
    if action.kind is ActionKind.GET:
        result = fhir.get(action.url or "")
        if result.ok:
            return GET_OBSERVATION.format(data=result.data), None
        return GET_ERROR.format(error=result.error), None

    if action.kind is ActionKind.POST:
        check = check_post_payload(action.post_body or "")
        # Upstream branches only on JSON validity; our resource-type check is recorded
        # but must not change the observation, or parity breaks.
        return (POST_ACCEPTED if check.json_valid else POST_INVALID), check

    return None, None


def _post_check_dict(check: PostCheck | None) -> dict | None:
    if check is None:
        return None
    return {
        "json_valid": check.json_valid,
        "resource_type": check.resource_type,
        "resource_allowed": check.resource_allowed,
        "schema_valid": check.schema_valid,
        "error": check.error,
    }


__all__ = [
    "run_episode",
    "canonical_action",
    "STATUS_COMPLETED",
    "STATUS_INVALID_ACTION",
    "STATUS_LIMIT_REACHED",
    "STATUS_CONTEXT_LIMIT",
    "STATUS_ERROR",
]
