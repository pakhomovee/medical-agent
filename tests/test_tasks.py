"""Template analysis -- the evidence behind gate G1.

The masking must be conservative: over-masking merges genuinely distinct templates and
would make G1 look better than it is, which is the failure mode that matters here.
"""

from __future__ import annotations

import json

import pytest

from uqma.envs.medagentbench import tasks as T

DATA = "data/test_data_v2.json"


def _write(tmp_path, items):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_mask_collapses_mrn_date_number_and_quote():
    a = T.mask("What's the most recent magnesium level of the patient S3032536 within last 24 hours?")
    b = T.mask("What's the most recent magnesium level of the patient S6315806 within last 48 hours?")
    assert a == b
    assert "<MRN>" in a and "<NUM>" in a


def test_mask_normalises_curly_apostrophes():
    # The corpus mixes ' and '; without NFKC these split one template into two.
    assert T.mask("What's the age") == T.mask("What’s the age")


def test_mask_does_not_merge_distinct_instructions():
    a = T.mask("What's the age of the patient with MRN of S1?")
    b = T.mask("What's the most recent CBG of the patient S1?")
    assert a != b


def test_datetime_masked_before_its_component_numbers():
    masked = T.mask("It's 2023-11-13T10:15:00+00:00 now.")
    assert "<DATETIME>" in masked
    assert "2023" not in masked


def test_category_and_instance_split():
    task = T.Task(id="task10_27", instruction="x", context="", sol=None, eval_mrn="S1")
    assert task.category == "task10"
    assert task.instance == "27"


def test_templates_group_by_id_prefix(tmp_path):
    items = [
        {"id": "task1_1", "instruction": "Find MRN for S1", "context": "", "sol": ["a"]},
        {"id": "task1_2", "instruction": "Find MRN for S2", "context": "", "sol": ["b"]},
        {"id": "task2_1", "instruction": "Order a referral for S3", "context": "", "sol": None},
    ]
    groups = T.templates(T.load_tasks(_write(tmp_path, items)))
    assert [g.key for g in groups] == ["task1", "task2"]
    assert [g.n_instances for g in groups] == [2, 1]


def test_action_classification_uses_sibling_vote(tmp_path):
    items = [
        {"id": "task8_1", "instruction": "Order orthopedic referral for S1", "context": "", "sol": None},
        {"id": "task8_2", "instruction": "Order orthopedic referral for S2", "context": "", "sol": None},
        {"id": "task7_1", "instruction": "What is the most recent CBG of S3?", "context": "", "sol": None},
    ]
    groups = {g.key: g for g in T.templates(T.load_tasks(_write(tmp_path, items)))}
    assert groups["task8"].is_action
    assert not groups["task7"].is_action


def test_agreement_flags_a_category_split_across_masks(tmp_path):
    items = [
        {"id": "task1_1", "instruction": "Find the MRN for S1", "context": "", "sol": None},
        {"id": "task1_2", "instruction": "Compute the average CBG for S2", "context": "", "sol": None},
    ]
    report = T.agreement(T.load_tasks(_write(tmp_path, items)))
    assert not report["consistent"]
    assert "task1" in report["categories_split_across_masks"]


def test_agreement_flags_a_mask_spanning_categories(tmp_path):
    # Real MRNs are S + 7 digits; short ids like "S1" are deliberately not masked.
    items = [
        {"id": "task1_1", "instruction": "Find the MRN for S1234567", "context": "", "sol": None},
        {"id": "task2_1", "instruction": "Find the MRN for S2345678", "context": "", "sol": None},
    ]
    report = T.agreement(T.load_tasks(_write(tmp_path, items)))
    assert not report["consistent"]
    assert report["masks_spanning_categories"]


def test_missing_eval_mrn_is_tolerated(tmp_path):
    # task1_29 and task1_30 in v2 are "patient not found" cases with no eval_MRN.
    items = [{"id": "task1_29", "instruction": "Find MRN", "context": "", "sol": ["Patient not found"]}]
    tasks = T.load_tasks(_write(tmp_path, items))
    assert tasks[0].eval_mrn is None
    assert T.distinct_patients(tasks) == set()


def test_load_rejects_a_non_list_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"id": "task1_1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON list"):
        T.load_tasks(path)


@pytest.mark.skipif(not __import__("pathlib").Path(DATA).exists(), reason="task data absent")
def test_real_v2_corpus_has_ten_templates_five_of_them_actions():
    """Pins the G1 result so a data swap cannot silently change the power analysis."""
    tasks = T.load_tasks(DATA)
    groups = T.templates(tasks)
    assert len(tasks) == 300
    assert len(groups) == 10
    assert all(g.n_instances == 30 for g in groups)
    assert sum(g.is_action for g in groups) == 5
    assert T.agreement(tasks)["consistent"]
