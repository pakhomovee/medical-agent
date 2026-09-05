"""Episode semantics, including the upstream behaviours G4 depends on."""

from __future__ import annotations

import json

import pytest

from uqma.agent.loop import (
    STATUS_COMPLETED,
    STATUS_CONTEXT_LIMIT,
    STATUS_INVALID_ACTION,
    STATUS_LIMIT_REACHED,
    run_episode,
)
from uqma.envs.medagentbench.fhir import FhirClient, GetResult
from uqma.envs.medagentbench.tasks import Task
from uqma.inference.base import Generation
from uqma.inference.stub import ScriptedBackend

FUNCTIONS = [{"name": "GET {api_base}/Observation", "description": "..."}]


class FakeFhir(FhirClient):
    """FhirClient with the network replaced, so tests need no server."""

    def __init__(self, payload=None, ok=True):
        super().__init__("http://fake/fhir")
        self._payload = payload if payload is not None else {"resourceType": "Bundle", "total": 0}
        self._ok = ok
        self.calls: list[str] = []

    def get(self, url: str) -> GetResult:
        self.calls.append(url)
        if not self._ok:
            return GetResult(ok=False, error="connection refused")
        return GetResult(ok=True, data=self._payload)

    def verify(self) -> bool:
        return True


@pytest.fixture
def task():
    return Task(
        id="task5_1",
        instruction="Check patient S1 magnesium. If low, order replacement.",
        context="It's 2023-11-13T10:15:00+00:00 now.",
        sol=None,
        eval_mrn="S1",
    )


def run(task, responses, fhir=None, max_round=5):
    return run_episode(
        task, ScriptedBackend(responses), fhir or FakeFhir(), FUNCTIONS, max_round=max_round
    )


def test_finish_completes_and_captures_result(task):
    episode = run(task, ["GET http://fake/fhir/Observation?patient=S1", "FINISH([1.2])"])
    assert episode.status == STATUS_COMPLETED
    assert episode.result == "[1.2]"
    assert episode.n_turns == 2


def test_unrecognised_action_terminates_episode(task):
    episode = run(task, ["Let me think about this first."])
    assert episode.status == STATUS_INVALID_ACTION
    assert episode.n_turns == 1


def test_malformed_post_does_not_terminate(task):
    # Upstream injects "Invalid POST request" and keeps going; only a bad prefix stops it.
    episode = run(task, ["POST http://fake/fhir/Observation\n{not json", "FINISH([])"])
    assert episode.status == STATUS_COMPLETED
    assert episode.n_turns == 2
    assert episode.turns[0].post_check["json_valid"] is False
    assert "Invalid POST request" in episode.turns[0].observation


def test_round_limit_reached(task):
    episode = run(task, ["GET http://fake/fhir/Observation?p=1"] * 10, max_round=3)
    assert episode.status == STATUS_LIMIT_REACHED
    assert episode.n_turns == 3


def test_context_limit_is_not_an_error(task):
    class Overflowing:
        def generate(self, messages, capture_logprobs=True, **kwargs):
            return Generation(text="", truncated_by_context=True)

    episode = run_episode(task, Overflowing(), FakeFhir(), FUNCTIONS)
    assert episode.status == STATUS_CONTEXT_LIMIT
    assert episode.n_turns == 0


def test_get_error_is_surfaced_to_the_model(task):
    episode = run(task, ["GET http://fake/fhir/Observation?p=1", "FINISH([])"],
                  fhir=FakeFhir(ok=False))
    assert "Error in sending the GET request" in episode.turns[0].observation
    assert episode.status == STATUS_COMPLETED


def test_schema_valid_post_is_counted(task):
    payload = {"resourceType": "Observation", "status": "final"}
    episode = run(task, [f"POST http://fake/fhir/Observation\n{json.dumps(payload)}", "FINISH([])"])
    assert episode.n_post_attempts == 1
    assert episode.n_schema_valid_posts == 1


def test_post_with_unwritable_resource_is_json_valid_but_not_schema_valid(task):
    # Parity: upstream accepts any JSON, so the observation must still be the success
    # string. Our stricter resource check is recorded separately.
    payload = {"resourceType": "Patient", "id": "S1"}
    episode = run(task, [f"POST http://fake/fhir/Patient\n{json.dumps(payload)}", "FINISH([])"])
    check = episode.turns[0].post_check
    assert check["json_valid"] is True
    assert check["resource_allowed"] is False
    assert episode.n_schema_valid_posts == 0
    assert "accepted and executed successfully" in episode.turns[0].observation


def test_history_alternates_and_ends_with_assistant_on_finish(task):
    episode = run(task, ["GET http://fake/fhir/Observation?p=1", "FINISH([])"])
    roles = [m["role"] for m in episode.history]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_opening_message_is_a_user_message(task):
    # Upstream injects the whole prompt with role="user", not as a system message.
    episode = run(task, ["FINISH([])"])
    assert episode.history[0]["role"] == "user"
    assert "You are an expert in using FHIR functions" in episode.history[0]["content"]


def test_turn_records_carry_logprobs_and_prefix_source(task):
    episode = run(task, ["FINISH([])"])
    turn = episode.turns[0]
    assert turn.token_logprobs is not None and len(turn.token_logprobs) > 0
    assert turn.prefix_source == "rollout"


def test_backend_exception_becomes_task_error(task):
    class Broken:
        def generate(self, messages, capture_logprobs=True, **kwargs):
            raise RuntimeError("server died")

    episode = run_episode(task, Broken(), FakeFhir(), FUNCTIONS)
    assert episode.status == "task_error"
    assert "server died" in episode.error
