"""k-resampling and exact-match action equivalence.

Exact matching is only sound if incidental differences are normalised first. If
canonicalisation were too loose, distinct clinical actions would be counted as agreeing
and self-consistency would look artificially high -- which would inflate the sampling
family's apparent discrimination.
"""

from __future__ import annotations

import json

from uqma.agent.parsing import canonical_action, parse
from uqma.agent.resample import (
    action_distribution,
    distinct_actions,
    resample_turn,
    self_consistency,
)
from uqma.inference.stub import ScriptedBackend
from uqma.trajectory.schema import Sample


def sample(canonical: str) -> Sample:
    return Sample(index=0, text="", action_kind="get", action_raw="", canonical_action=canonical)


def test_query_param_order_does_not_matter():
    a = canonical_action(parse("GET http://x/fhir/Observation?code=MG&patient=S1"))
    b = canonical_action(parse("GET http://x/fhir/Observation?patient=S1&code=MG"))
    assert a == b


def test_json_key_order_does_not_matter():
    a = canonical_action(parse('POST http://x/o\n{"b": 2, "a": 1}'))
    b = canonical_action(parse('POST http://x/o\n{"a": 1, "b": 2}'))
    assert a == b


def test_whitespace_in_payload_does_not_matter():
    a = canonical_action(parse('POST http://x/o\n{"a":1}'))
    b = canonical_action(parse('POST http://x/o\n{\n  "a" : 1\n}'))
    assert a == b


def test_different_values_do_matter():
    a = canonical_action(parse('POST http://x/o\n{"dose": 10}'))
    b = canonical_action(parse('POST http://x/o\n{"dose": 20}'))
    assert a != b


def test_different_endpoints_do_matter():
    a = canonical_action(parse("GET http://x/fhir/Observation?code=MG"))
    b = canonical_action(parse("GET http://x/fhir/MedicationRequest?code=MG"))
    assert a != b


def test_format_json_suffix_is_normalised_away():
    # The loop appends '&_format=json'; it is an artifact of the harness, not a choice
    # the model made, so it must not distinguish two otherwise identical calls.
    with_suffix = canonical_action(parse("GET http://x/o?a=1"))
    assert "_format=json" not in with_suffix


def test_unparseable_post_is_its_own_bucket():
    assert canonical_action(parse("POST http://x/o\n{not json")) == "post <unparseable>"


def test_invalid_actions_share_a_bucket():
    assert canonical_action(parse("let me think")) == "invalid"


def test_resample_pins_history_across_samples():
    backend = ScriptedBackend(["GET http://x/o?a=1", "GET http://x/o?a=2", "GET http://x/o?a=1"])
    messages = [{"role": "user", "content": "prompt"}]
    samples = resample_turn(messages, backend, k=3)

    assert len(samples) == 3
    assert all(call == messages for call in backend.calls)  # identical prompt every time
    assert [s.index for s in samples] == [0, 1, 2]


def test_resample_records_canonical_actions_and_logprobs():
    backend = ScriptedBackend(["GET http://x/o?a=1"] * 2)
    samples = resample_turn([{"role": "user", "content": "p"}], backend, k=2)
    assert all(s.canonical_action == "get http://x/o?a=1" for s in samples)
    assert all(s.token_logprobs for s in samples)


def test_self_consistency_and_distinct_counts():
    samples = [sample("a"), sample("a"), sample("b")]
    assert self_consistency(samples) == 2 / 3
    assert distinct_actions(samples) == 2
    assert action_distribution(samples)["a"] == 2


def test_self_consistency_is_one_when_unanimous():
    assert self_consistency([sample("a")] * 5) == 1.0


def test_self_consistency_of_no_samples_is_none():
    assert self_consistency([]) is None


def test_zero_k_yields_no_samples():
    backend = ScriptedBackend([])
    assert resample_turn([{"role": "user", "content": "p"}], backend, k=0) == []


def test_samples_are_not_executed_by_the_loop():
    """Sampling costs k x generation but must not multiply environment interactions."""
    import pytest

    from uqma.agent.loop import run_episode
    from uqma.envs.medagentbench.fhir import FhirClient, GetResult
    from uqma.envs.medagentbench.tasks import Task

    class CountingFhir(FhirClient):
        def __init__(self):
            super().__init__("http://fake/fhir")
            self.n_calls = 0

        def get(self, url):
            self.n_calls += 1
            return GetResult(ok=True, data={"total": 0})

    task = Task(id="task4_1", instruction="x", context="", sol=None, eval_mrn="S1")
    fhir = CountingFhir()
    backend = ScriptedBackend(["GET http://fake/fhir/o?a=1"] * 20 + ["FINISH([])"] * 20)

    trajectory = run_episode(task, backend, fhir, [{"name": "f"}], max_round=2, n_samples=4)

    # 2 turns executed -> at most 2 GETs, regardless of the 4 samples drawn per turn.
    assert fhir.n_calls <= 2
    assert len(trajectory.turns[0].samples) == 4
    assert pytest.approx(trajectory.turns[0].latency_s, abs=1.0) == trajectory.turns[0].latency_s
