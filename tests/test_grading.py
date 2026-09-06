"""Grader mutation tests — the suite plan §6.4 calls the most important.

Every number in the thesis rests on these labels, so the graders need to be shown to
accept a correct payload and reject each way of getting it wrong. A grader that returns
False for everything would look exactly like a weak model.

Requires ``data/refsol.py`` (gitignored answer key, Box link in the upstream README).
Skipped without it, so CI on a clean checkout still passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uqma.envs.medagentbench.grading import (
    AGENT_ROLE,
    GradingInput,
    Message,
    RefsolGrader,
    grade_task,
    to_agent_history,
)
from uqma.envs.medagentbench.prompts import POST_ACCEPTED

REFSOL = Path("data/refsol.py")
TASKS = Path("data/test_data_v2.json")
API_BASE = "http://localhost:8080/fhir"

requires_refsol = pytest.mark.skipif(
    not REFSOL.exists(), reason="data/refsol.py absent (gitignored answer key)"
)


# --- history adaptation (no refsol needed) -------------------------------------------

def test_history_is_translated_to_upstream_shape():
    """Upstream graders read .role/.content attributes and spell the assistant 'agent'."""
    converted = to_agent_history(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    )
    assert all(isinstance(m, Message) for m in converted)
    assert [m.role for m in converted] == ["user", AGENT_ROLE]
    assert converted[1].content == "a"


def test_user_role_is_left_alone():
    assert to_agent_history([{"role": "user", "content": "x"}])[0].role == "user"


def test_missing_content_becomes_empty_string():
    assert to_agent_history([{"role": "user"}])[0].content == ""


def test_grade_task_without_a_grader_returns_none_not_false():
    assert grade_task(None, {"id": "task3_1"}, GradingInput(result="[]")) is None


# --- grader behaviour (needs refsol) --------------------------------------------------

@pytest.fixture(scope="module")
def grader():
    return RefsolGrader(REFSOL, API_BASE)


@pytest.fixture(scope="module")
def cases():
    return {t["id"]: t for t in json.loads(TASKS.read_text(encoding="utf-8"))}


def _post_trajectory(payload: dict, resource: str = "Observation"):
    """A completed trajectory whose single POST carries ``payload``."""
    return GradingInput(
        result="[]",
        history=to_agent_history([
            {"role": "user", "content": "<prompt>"},
            {"role": "assistant",
             "content": f"POST {API_BASE}/{resource}\n{json.dumps(payload)}"},
            {"role": "user", "content": POST_ACCEPTED},
            {"role": "assistant", "content": "FINISH([])"},
        ]),
        status="completed",
    )


def _gold_task3(mrn: str) -> dict:
    return {
        "resourceType": "Observation",
        "category": [{"coding": [{"system": "http://hl7.org/fhir/observation-category",
                                  "code": "vital-signs", "display": "Vital Signs"}]}],
        "code": {"text": "BP"},
        "effectiveDateTime": "2023-11-13T10:15:00+00:00",
        "status": "final",
        "valueString": "118/77 mmHg",
        "subject": {"reference": f"Patient/{mrn}"},
    }


@requires_refsol
def test_all_ten_graders_are_present(grader):
    assert set(grader.available_categories()) == {f"task{i}" for i in range(1, 11)}


@requires_refsol
def test_correct_payload_grades_true(grader, cases):
    """The load-bearing assertion: graders can return True through our adapter."""
    case = cases["task3_1"]
    assert grade_task(grader, case, _post_trajectory(_gold_task3(case["eval_MRN"]))) is True


@pytest.mark.parametrize(
    "mutation,description",
    [
        ({"valueString": "999/999 mmHg"}, "wrong measured value"),
        ({"subject": {"reference": "Patient/S0000000"}}, "wrong patient"),
        ({"code": {"text": "HR"}}, "wrong observation code"),
        ({"status": "preliminary"}, "wrong status"),
        ({"effectiveDateTime": "2020-01-01T00:00:00+00:00"}, "wrong timestamp"),
        ({"resourceType": "MedicationRequest"}, "wrong resource type"),
        ({"category": []}, "missing category"),
    ],
)
@requires_refsol
def test_each_mutation_is_rejected(grader, cases, mutation, description):
    case = cases["task3_1"]
    payload = _gold_task3(case["eval_MRN"]) | mutation
    assert grade_task(grader, case, _post_trajectory(payload)) is False, description


@requires_refsol
def test_missing_required_field_is_rejected(grader, cases):
    case = cases["task3_1"]
    payload = _gold_task3(case["eval_MRN"])
    del payload["valueString"]
    assert grade_task(grader, case, _post_trajectory(payload)) is False


@requires_refsol
def test_two_posts_are_rejected_even_when_one_is_correct(grader, cases):
    """A documented label-noise mode: task3 requires exactly one POST.

    An agent that posts twice fails even if one payload is right. Worth carrying into
    the κ audit alongside the noise modes the v2 authors documented.
    """
    case = cases["task3_1"]
    gold = json.dumps(_gold_task3(case["eval_MRN"]))
    output = GradingInput(
        result="[]",
        history=to_agent_history([
            {"role": "user", "content": "<prompt>"},
            {"role": "assistant", "content": f"POST {API_BASE}/Observation\n{gold}"},
            {"role": "user", "content": POST_ACCEPTED},
            {"role": "assistant", "content": f"POST {API_BASE}/Observation\n{gold}"},
            {"role": "user", "content": POST_ACCEPTED},
            {"role": "assistant", "content": "FINISH([])"},
        ]),
        status="completed",
    )
    assert grade_task(grader, case, output) is False


@requires_refsol
def test_unfinished_episode_is_incorrect_without_calling_the_grader(grader, cases):
    assert grade_task(grader, cases["task3_1"], GradingInput(result=None)) is False


@requires_refsol
def test_refsol_relative_import_shim_works(grader):
    """refsol opens with `from .utils import *`; loading it standalone needs a shim."""
    import sys

    assert "uqma_refsol_pkg.utils" in sys.modules
    assert callable(sys.modules["uqma_refsol_pkg.utils"].send_get_request)
