"""Gate G2 arithmetic and the grading-unavailable path.

The load-bearing property is that a missing ``refsol.py`` yields INCOMPLETE, never a
reported 0% success rate. Conflating "could not grade" with "graded and wrong" would
silently fail every candidate model and send the project down the wrong branch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from g2_model_gate import (  # noqa: E402
    ACTION_SR_THRESHOLD,
    SCHEMA_VALID_THRESHOLD,
    evaluate_gate,
    select,
    summarise,
)
from uqma.envs.medagentbench.grading import GradingInput, grade_task  # noqa: E402
from uqma.envs.medagentbench.tasks import Task  # noqa: E402


def row(category, is_action, correct, status="completed", posts=1, valid=1, turns=3):
    return {
        "task_id": f"{category}_1",
        "category": category,
        "is_action": is_action,
        "status": status,
        "correct": correct,
        "n_turns": turns,
        "n_post_attempts": posts,
        "n_schema_valid_posts": valid,
        "n_invalid_action_turns": 0,
        "n_generated_tokens": 50,
        "wall_s": 1.0,
    }


def test_action_and_query_rates_are_computed_separately():
    rows = [
        row("task8", True, True), row("task8", True, False),
        row("task2", False, True), row("task2", False, True),
    ]
    s = summarise(rows)
    assert s["action_sr"] == 0.5
    assert s["query_sr"] == 1.0
    assert s["overall_sr"] == 0.75


def test_ungraded_rows_do_not_count_as_wrong():
    rows = [row("task8", True, None), row("task8", True, True)]
    s = summarise(rows)
    assert s["n_graded"] == 1
    assert s["action_sr"] == 1.0  # not 0.5


def test_all_ungraded_yields_none_not_zero():
    s = summarise([row("task8", True, None), row("task2", False, None)])
    assert s["action_sr"] is None
    assert s["overall_sr"] is None


def test_schema_valid_rate_over_post_attempts():
    rows = [row("task8", True, True, posts=2, valid=1), row("task5", True, True, posts=2, valid=2)]
    assert summarise(rows)["schema_valid_rate"] == pytest.approx(0.75)


def test_schema_valid_rate_is_none_when_no_posts_attempted():
    assert summarise([row("task2", False, True, posts=0, valid=0)])["schema_valid_rate"] is None


def test_gate_requires_both_thresholds():
    strong = summarise([row("task8", True, True)] * 10)
    assert evaluate_gate(strong, graded=True)["gate"] == "PASS"

    weak_action = summarise([row("task8", True, False)] * 10)
    assert evaluate_gate(weak_action, graded=True)["gate"] == "FAIL"

    bad_schema = summarise([row("task8", True, True, posts=10, valid=1)] * 10)
    result = evaluate_gate(bad_schema, graded=True)
    assert result["gate"] == "FAIL"
    assert result["action_sr_pass"] and not result["schema_valid_pass"]


def test_gate_is_incomplete_without_a_grader():
    s = summarise([row("task8", True, None)] * 5)
    result = evaluate_gate(s, graded=False)
    assert result["gate"] == "INCOMPLETE"
    assert "refsol" in result["reason"]


def test_thresholds_match_the_plan():
    assert ACTION_SR_THRESHOLD == 0.40
    assert SCHEMA_VALID_THRESHOLD == 0.80


def test_gate_is_decided_on_action_sr_not_overall_sr():
    # The Gemma2 failure mode: strong on query tasks, 0% on action tasks.
    rows = [row("task8", True, False)] * 10 + [row("task2", False, True)] * 40
    s = summarise(rows)
    assert s["overall_sr"] == 0.8
    assert s["action_sr"] == 0.0
    assert evaluate_gate(s, graded=True)["gate"] == "FAIL"


def test_grade_task_returns_none_without_grader():
    assert grade_task(None, {"id": "task8_1"}, GradingInput(result="[]")) is None


def test_grade_task_counts_unfinished_episodes_as_incorrect():
    class Grader:
        def grade(self, case_data, output):
            raise AssertionError("must not be called when result is None")

    assert grade_task(Grader(), {"id": "task8_1"}, GradingInput(result=None)) is False


def test_select_stratifies_across_templates():
    tasks = [
        Task(id=f"task{c}_{i}", instruction="x", context="", sol=None, eval_mrn="S1")
        for c in range(1, 11)
        for i in range(1, 31)
    ]
    chosen = select(tasks, per_template=2, limit=None)
    assert len(chosen) == 20
    assert len({t.category for t in chosen}) == 10


def test_select_limit_alone_would_not_stratify():
    # Documents why --per-template exists: the file is ordered by category, so a plain
    # head slice returns only task1.
    tasks = [
        Task(id=f"task{c}_{i}", instruction="x", context="", sol=None, eval_mrn="S1")
        for c in range(1, 11)
        for i in range(1, 31)
    ]
    assert len({t.category for t in select(tasks, None, limit=20)}) == 1


def test_per_template_summary_marks_action_templates():
    rows = [row("task8", True, True), row("task2", False, True)]
    per = summarise(rows)["per_template"]
    assert per["task8"]["is_action"] is True
    assert per["task2"]["is_action"] is False
