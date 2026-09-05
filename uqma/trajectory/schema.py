"""The trajectory log schema — the load-bearing interface of the project.

Stage 1 (GPU) writes these; stages 2-6 read them and never touch a model (plan §6.2).
That separation is what makes weeks 21-27 runnable on a laptop, and it only holds if the
schema is stable. So:

* ``SCHEMA_VERSION`` is stamped into every run manifest. Bump it for any change that is
  not purely additive-with-a-default, and write a migration rather than editing readers
  to cope with both shapes.
* Large arrays never live inline. Hidden states and attention are written as separate
  artifacts and referenced by ``HiddenStateRef``, because a sweep produces ~1-2 TB of
  them and the tabular log must stay small enough to load whole.
* Fields the plan anticipates but does not yet populate exist now with defaults --
  ``prefix_source`` (plan §4.3) and ``samples`` (§6.4 k-resampling) among them. Adding a
  field later means migrating a terabyte of logs; adding it now costs a default value.

Round-tripping is enforced by tests: ``from_dict(to_dict(x)) == x`` for every type here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0.0"


class PrefixSource(str, Enum):
    """Where the history preceding a turn came from.

    ``ROLLOUT`` is the model's own trajectory — the only value produced today. The others
    exist for prefix seeding (plan §4.3), which is a contingency, and for correct-and-
    continue deferral (§4.2), where an oracle gold action is injected mid-trajectory.
    Analysis must be able to filter on this: an estimator measured off-policy is not
    measuring the deployment distribution unless continuation semantics are in force.
    """

    ROLLOUT = "rollout"
    GOLD = "gold"
    MODEL = "model"
    INJECTED = "injected"


class EpisodeStatus(str, Enum):
    """Mirrors upstream ``SampleStatus`` so gate G4 can compare per-task outcomes."""

    COMPLETED = "completed"
    AGENT_INVALID_ACTION = "agent_invalid_action"
    TASK_LIMIT_REACHED = "task_limit_reached"
    AGENT_CONTEXT_LIMIT = "agent_context_limit"
    TASK_ERROR = "task_error"


@dataclass
class HiddenStateRef:
    """Pointer to hidden states written by stage 2, never the states themselves.

    Stage 2 is a separate pass over the stage-1 log (plan §6.2), so this is filled in
    after the fact. Changing which layers you extract then costs one cheap re-run instead
    of regenerating every trajectory.
    """

    path: str
    layer_indices: list[int] = field(default_factory=list)
    token_start: int | None = None
    token_end: int | None = None
    dtype: str = "bfloat16"


@dataclass
class Sample:
    """One of k actions resampled at a turn with history held fixed.

    The sampling protocol resamples the action at turn t, not whole trajectories, which
    keeps cost at k x rather than combinatorial (proposal §6.3). Because tool calls are
    structured, semantic equivalence is exact — two samples match iff they are the same
    call with the same arguments — which is what makes semantic entropy cheap here.
    """

    index: int
    text: str
    action_kind: str
    action_raw: str
    token_logprobs: list[float] | None = None
    n_generated_tokens: int | None = None
    finish_reason: str | None = None
    canonical_action: str | None = None
    correct: bool | None = None


@dataclass
class Turn:
    """One assistant turn: what the model saw, what it emitted, what came back."""

    index: int
    generation: str
    action_kind: str
    action_raw: str
    observation: str | None = None
    url: str | None = None
    post_check: dict[str, Any] | None = None
    token_logprobs: list[float] | None = None
    n_generated_tokens: int | None = None
    finish_reason: str | None = None
    samples: list[Sample] = field(default_factory=list)
    hidden_states: HiddenStateRef | None = None
    prefix_source: str = PrefixSource.ROLLOUT.value
    gated: bool = False
    injected_action: str | None = None
    correct: bool | None = None
    error_class: str | None = None
    latency_s: float = 0.0

    @property
    def is_commitment(self) -> bool:
        """A turn that emits an irreversible write.

        Irreversibility is a property of the deployment being modelled, not of this
        benchmark: MedAgentBench validates POST payloads and never sends them (plan
        §3 R4). The clinical review being modelled happens before an order is signed,
        which is exactly this point in the trajectory.
        """
        return self.action_kind == "post"


@dataclass
class Trajectory:
    task_id: str
    category: str
    status: str
    result: str | None = None
    turns: list[Turn] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    correct: bool | None = None
    error: str | None = None
    wall_s: float = 0.0
    seed: int | None = None
    run_id: str | None = None

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def prompt_for_turn(self, index: int) -> list[dict[str, str]]:
        """Reconstruct exactly what the model saw before turn ``index``.

        Stored as a slice of ``history`` rather than per-turn, because a per-turn copy
        duplicates every prior FHIR response once per subsequent turn -- O(n^2) in the
        turn count, and FHIR bundles are large. The transcript alternates
        user/assistant from the opening prompt, so turn i saw ``history[:2i+1]``.

        This stays correct under correct-and-continue deferral (plan §4.2): when a gold
        action is injected, ``history`` records the injected action, which is what later
        turns actually conditioned on, while ``Turn.generation`` keeps what the model
        itself emitted.
        """
        if index < 0 or index >= len(self.turns):
            raise IndexError(f"turn {index} out of range (have {len(self.turns)})")
        return self.history[: 2 * index + 1]

    @property
    def commitments(self) -> list[Turn]:
        return [t for t in self.turns if t.is_commitment]

    @property
    def n_post_attempts(self) -> int:
        return len(self.commitments)

    @property
    def n_schema_valid_posts(self) -> int:
        return sum(
            1 for t in self.commitments if t.post_check and t.post_check.get("schema_valid")
        )

    def to_dict(self) -> dict:
        return to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> Trajectory:
        return from_dict(cls, payload)


# --- (de)serialisation -------------------------------------------------------------
# Hand-rolled rather than pydantic: the dependency is not installed on the analysis box,
# and the schema is small enough that an explicit converter is clearer than a framework.

_SCALARS = (str, int, float, bool, type(None))


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if value is None and f.name in _OPTIONAL_OMITTED:
                continue
            out[f.name] = to_dict(value)
        return out
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, _SCALARS):
        return obj
    raise TypeError(f"cannot serialise {type(obj).__name__}")


# Omitted when None purely to keep logs small; readers restore the default.
_OPTIONAL_OMITTED = frozenset({"hidden_states", "injected_action", "error_class"})

_TYPE_MAP: dict[str, type] = {}


def _resolve(cls: type, name: str) -> type | None:
    return _TYPE_MAP.get(f"{cls.__name__}.{name}")


def from_dict(cls: type, payload: dict) -> Any:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown field(s) {sorted(unknown)}. This usually means the "
            f"log was written by a different SCHEMA_VERSION than {SCHEMA_VERSION}; write a "
            "migration rather than loosening this check."
        )
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in payload:
            continue
        value = payload[f.name]
        nested = _resolve(cls, f.name)
        if nested is None or value is None:
            kwargs[f.name] = value
        elif isinstance(value, list):
            kwargs[f.name] = [from_dict(nested, v) for v in value]
        else:
            kwargs[f.name] = from_dict(nested, value)
    return cls(**kwargs)


_TYPE_MAP.update(
    {
        "Turn.samples": Sample,
        "Turn.hidden_states": HiddenStateRef,
        "Trajectory.turns": Turn,
    }
)
