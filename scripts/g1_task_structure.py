#!/usr/bin/env python3
"""Gate G1 -- task structure and grader analysis.

Answers the question that most changes the thesis (plan §3 R2): are the tasks
parameterised templates over a patient pool, and can the graders be inverted to yield a
gold action for the correct-and-continue deferral design (plan §4.2)?

Runs on CPU in seconds. No GPU, no Docker, no FHIR server, no model. Do this first.

    python scripts/g1_task_structure.py --tasks data/test_data_v1.json
    python scripts/g1_task_structure.py --tasks data/test_data_v1.json \\
        --refsol data/refsol.py --out runs/g1_v1.json

``refsol.py`` is optional; without it the grader half is skipped and reported as
unavailable rather than silently passing.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uqma.envs.medagentbench import tasks as T  # noqa: E402

# Field names whose presence in a grader indicates payload-level checking rather than a
# single aggregate boolean -- the distinction that decides how much of plan §6.2's
# "field-level graded labels" is new work versus already shipped.
_FIELD_MARKERS = (
    "resourceType", "subject", "reference", "valueQuantity", "code", "coding",
    "status", "intent", "authoredOn", "effectiveDateTime", "dosageInstruction",
    "medicationCodeableConcept", "occurrenceDateTime", "system", "value", "unit",
)

# A grader that reads the conversation history is reading the POST payload out of the
# transcript -- which is the only place it exists, since POSTs are never sent. That is
# also the signal that a gold action can be reconstructed (invertibility).
_HISTORY_MARKERS = ("history", "result", "results")


def analyse_tasks(task_path: Path) -> dict:
    tasks = T.load_tasks(task_path)
    groups = T.templates(tasks)
    counts = T.instances_per_template(tasks)
    patients = T.distinct_patients(tasks)

    action = [g for g in groups if g.is_action]
    query = [g for g in groups if not g.is_action]
    conditional = [g for g in groups if g.is_conditional]

    return {
        "file": str(task_path),
        "n_tasks": len(tasks),
        "n_templates": len(groups),
        "instances_per_template": dict(sorted(counts.items())),
        "n_distinct_patients": len(patients),
        "n_tasks_with_sol": sum(1 for t in tasks if t.has_solution),
        "templates": [
            {
                "key": g.key,
                "n_instances": g.n_instances,
                "is_action": g.is_action,
                "is_conditional": g.is_conditional,
                "masked_instruction": g.masked_instruction[:200],
            }
            for g in groups
        ],
        "n_action_templates": len(action),
        "n_query_templates": len(query),
        "n_conditional_templates": len(conditional),
        "n_action_instances": sum(g.n_instances for g in action),
        "template_agreement": T.agreement(tasks),
    }


def analyse_refsol(refsol_path: Path) -> dict:
    """Static analysis of the grader module. Parsed, never executed."""
    source = refsol_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    graders: dict[str, dict] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        segment = ast.get_source_segment(source, node) or ""
        graders[node.name] = {
            "n_lines": (node.end_lineno or node.lineno) - node.lineno + 1,
            "args": [a.arg for a in node.args.args],
            "field_markers": sorted({m for m in _FIELD_MARKERS if m in segment}),
            "reads_history": any(f".{m}" in segment for m in _HISTORY_MARKERS),
            "queries_fhir": "requests" in segment or "send_get_request" in segment,
        }

    task_graders = {k: v for k, v in graders.items() if k.startswith("task")}
    with_fields = [k for k, v in task_graders.items() if v["field_markers"]]
    with_history = [k for k, v in task_graders.items() if v["reads_history"]]

    return {
        "file": str(refsol_path),
        "n_functions": len(graders),
        "n_task_graders": len(task_graders),
        "grader_names": sorted(task_graders),
        "per_category_not_per_task": all("_" not in k for k in task_graders),
        "field_level_graders": sorted(with_fields),
        "n_field_level": len(with_fields),
        "history_reading_graders": sorted(with_history),
        "n_history_reading": len(with_history),
        "graders": task_graders,
    }


def verdict(task_report: dict, refsol_report: dict | None) -> dict:
    """Decide G1 and spell out what each finding implies for the plan."""
    templated = (
        task_report["n_templates"] > 0
        and task_report["n_tasks"] > task_report["n_templates"]
        and task_report["template_agreement"]["consistent"]
    )
    n_action_templates = task_report["n_action_templates"]

    notes: list[str] = []
    if templated:
        notes.append(
            f"Templating CONFIRMED: {task_report['n_tasks']} tasks over "
            f"{task_report['n_templates']} templates; id-grouping and mask-grouping agree."
        )
    else:
        notes.append(
            "Templating NOT confirmed -- id-grouping and mask-grouping disagree, or every "
            "task is unique. Invoke the power fallback (plan §3 R2) and drop RQ3 now."
        )

    notes.append(
        f"Effective clusters for action-task analysis: {n_action_templates} "
        f"({task_report['n_action_instances']} instances). Task-clustered bootstrap "
        "resamples templates, so this -- not the instance count -- bounds power."
    )
    if n_action_templates < 8:
        notes.append(
            "FEWER THAN 8 ACTION TEMPLATES: ranking estimator families is not achievable. "
            "State RQ1 as 'does any estimator beat chance, and by how much'. Note that "
            "test_data_v1.json is a strict SUBSET of test_data_v2.json, so combining them "
            "adds no clusters -- more templates would have to come from the PSB 2026 task "
            "set (ericoericochen/medagentbenchv2), which is a separate acquisition."
        )

    if task_report["n_tasks_with_sol"] < task_report["n_tasks"]:
        notes.append(
            f"Only {task_report['n_tasks_with_sol']}/{task_report['n_tasks']} tasks carry a "
            "'sol'. Grading is impossible without refsol.py -- do not report a success rate "
            "without it."
        )

    if refsol_report is None:
        notes.append(
            "refsol.py absent: grader invertibility UNRESOLVED. Download it from the Box "
            "link in the upstream README (and never commit it)."
        )
        invertible = None
    else:
        invertible = refsol_report["n_history_reading"] > 0 and refsol_report["n_field_level"] > 0
        notes.append(
            f"Graders: {refsol_report['n_task_graders']} task graders, "
            f"{refsol_report['n_field_level']} check payload fields, "
            f"{refsol_report['n_history_reading']} read the transcript."
        )
        notes.append(
            "Gold-action reconstruction looks FEASIBLE -- graders assert on payload fields, "
            "so the expected payload can be derived (plan §4.2)."
            if invertible
            else "Gold-action reconstruction looks HARD: graders do not assert on payload "
            "fields. Correct-and-continue may need hand-written gold actions per template."
        )

    return {
        "templating_confirmed": templated,
        "n_action_templates": n_action_templates,
        "grader_invertibility_feasible": invertible,
        "gate": "PASS" if templated else "FAIL",
        "notes": notes,
    }


def render(report: dict) -> str:
    t = report["tasks"]
    lines = [
        "=" * 78,
        f"GATE G1 -- TASK STRUCTURE   [{report['verdict']['gate']}]",
        "=" * 78,
        f"file                 {t['file']}",
        f"tasks                {t['n_tasks']}",
        f"templates            {t['n_templates']}  "
        f"(action {t['n_action_templates']}, query {t['n_query_templates']}, "
        f"conditional {t['n_conditional_templates']})",
        f"instances/template   {t['instances_per_template']}",
        f"action instances     {t['n_action_instances']}",
        f"distinct patients    {t['n_distinct_patients']}",
        f"tasks with 'sol'     {t['n_tasks_with_sol']}/{t['n_tasks']}",
        f"grouping agreement   {'consistent' if t['template_agreement']['consistent'] else 'INCONSISTENT'}",
        "",
        "TEMPLATES",
    ]
    for tpl in t["templates"]:
        flags = "".join(["A" if tpl["is_action"] else "q", "C" if tpl["is_conditional"] else "-"])
        lines.append(f"  {tpl['key']:<10} n={tpl['n_instances']:<4} [{flags}]  {tpl['masked_instruction'][:88]}")

    if report.get("refsol"):
        r = report["refsol"]
        lines += [
            "",
            "GRADERS (refsol.py, statically parsed)",
            f"  task graders       {r['n_task_graders']}  {r['grader_names']}",
            f"  per-category       {r['per_category_not_per_task']}",
            f"  field-level        {r['n_field_level']}  {r['field_level_graders']}",
            f"  read transcript    {r['n_history_reading']}  {r['history_reading_graders']}",
        ]

    lines += ["", "VERDICT"]
    lines += [f"  - {n}" for n in report["verdict"]["notes"]]
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate G1: task structure and graders")
    parser.add_argument("--tasks", required=True, type=Path, help="test_data_v*.json")
    parser.add_argument("--refsol", type=Path, default=None, help="refsol.py (optional)")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    if not args.tasks.exists():
        parser.error(f"tasks file not found: {args.tasks}")

    task_report = analyse_tasks(args.tasks)
    refsol_report = None
    if args.refsol:
        if args.refsol.exists():
            refsol_report = analyse_refsol(args.refsol)
        else:
            print(f"warning: refsol not found at {args.refsol}; skipping grader analysis",
                  file=sys.stderr)

    report = {
        "tasks": task_report,
        "refsol": refsol_report,
        "verdict": verdict(task_report, refsol_report),
    }
    print(render(report))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.out}")

    return 0 if report["verdict"]["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
