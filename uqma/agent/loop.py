"""The instrumented ReAct loop.

This replaces upstream's controller/worker/assigner machinery (plan §6.1): we need
per-token logprobs, k-resampling with history pinned, and a teacher-forced second pass,
none of which survive an HTTP boundary between us and the model.

Episode semantics are copied from upstream so that gate G4 can compare per-task outcomes:
``max_round`` defaults to 5 (upstream's default -- note the proposal text says 8, which
Spike A must resolve), an unrecognised action terminates the episode, and a malformed
POST body does not.

Instrumentation beyond upstream is additive and optional: turn records carry the raw
generation and, when the backend supplies them, token logprobs. Nothing here computes an
uncertainty score -- that is stage 4, off the logged artifact (plan §6.2).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

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
from .parsing import Action, ActionKind, parse

# Episode outcomes, named to match upstream's SampleStatus.
STATUS_COMPLETED = "completed"
STATUS_INVALID_ACTION = "agent_invalid_action"
STATUS_LIMIT_REACHED = "task_limit_reached"
STATUS_CONTEXT_LIMIT = "agent_context_limit"
STATUS_ERROR = "task_error"


@dataclass
class TurnRecord:
    """One assistant turn plus the observation it produced.

    ``prefix_source`` is reserved for prefix seeding (plan §4.3). It is always
    ``"rollout"`` today; the field exists now because adding it later means migrating a
    terabyte of logs.
    """

    index: int
    prompt_messages: list[dict]
    generation: str
    action_kind: str
    action_raw: str
    observation: str | None
    url: str | None = None
    post_check: dict | None = None
    logprobs: list[float] | None = None
    prefix_source: str = "rollout"
    latency_s: float = 0.0
    n_generated_tokens: int | None = None


@dataclass
class Episode:
    task_id: str
    category: str
    status: str
    result: str | None
    turns: list[TurnRecord] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    error: str | None = None
    wall_s: float = 0.0

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def n_post_attempts(self) -> int:
        return sum(1 for t in self.turns if t.action_kind == ActionKind.POST.value)

    @property
    def n_schema_valid_posts(self) -> int:
        return sum(
            1
            for t in self.turns
            if t.post_check is not None and t.post_check.get("schema_valid")
        )

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "wall_s": round(self.wall_s, 3),
            "n_turns": self.n_turns,
            "turns": [asdict(t) for t in self.turns],
            "history": self.history,
        }


def run_episode(
    task: Task,
    backend: Backend,
    fhir: FhirClient,
    functions: list[dict],
    max_round: int = 5,
    capture_logprobs: bool = True,
) -> Episode:
    """Run one task to termination and return the full record."""
    started = time.monotonic()
    opening = build_prompt(task, functions, fhir.api_base)
    history: list[dict] = [{"role": "user", "content": opening}]
    episode = Episode(
        task_id=task.id, category=task.category, status=STATUS_LIMIT_REACHED, result=None
    )

    try:
        for turn_index in range(max_round):
            turn_started = time.monotonic()
            generation: Generation = backend.generate(
                history, capture_logprobs=capture_logprobs
            )
            latency = time.monotonic() - turn_started

            if generation.truncated_by_context:
                episode.status = STATUS_CONTEXT_LIMIT
                break

            action = parse(generation.text)
            history.append({"role": "assistant", "content": generation.text})

            observation, post_check = _apply(action, fhir)

            episode.turns.append(
                TurnRecord(
                    index=turn_index,
                    prompt_messages=list(history[:-1]),
                    generation=generation.text,
                    action_kind=action.kind.value,
                    action_raw=action.raw,
                    observation=observation,
                    url=action.url,
                    post_check=_post_check_dict(post_check),
                    logprobs=generation.token_logprobs,
                    latency_s=round(latency, 3),
                    n_generated_tokens=generation.n_generated_tokens,
                )
            )

            if action.kind is ActionKind.FINISH:
                episode.status = STATUS_COMPLETED
                episode.result = action.finish_payload
                break
            if action.kind is ActionKind.INVALID:
                episode.status = STATUS_INVALID_ACTION
                break

            history.append({"role": "user", "content": observation})

    except Exception as exc:  # upstream also collapses any error into a failed episode
        episode.status = STATUS_ERROR
        episode.error = f"{type(exc).__name__}: {exc}"

    episode.history = history
    episode.wall_s = time.monotonic() - started
    return episode


def _apply(action: Action, fhir: FhirClient) -> tuple[str | None, PostCheck | None]:
    """Execute an action against the environment and return the observation text."""
    if action.kind is ActionKind.GET:
        result = fhir.get(action.url or "")
        if result.ok:
            return GET_OBSERVATION.format(data=result.data), None
        return GET_ERROR.format(error=result.error), None

    if action.kind is ActionKind.POST:
        check = check_post_payload(action.post_body or "")
        # Upstream branches only on JSON validity; the resource-type check is recorded
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
