#!/usr/bin/env python3
"""Gate G2 -- model selection gate.

Decides whether a candidate open-weight model can act competently enough that turn
labels are informative (plan §3 R1). Two published open-weight models score 0.00% on
MedAgentBench *action* tasks while scoring 8-39% on query tasks, so selection must be on
action success rate, never on overall success rate.

    G2 PASSES when action SR >= 40% AND schema-valid tool-call rate >= 80%.

Both thresholds must be measured under the scaffold we will actually ship, not upstream's:
scaffold quality alone moves success from 69.67% to 91% on the same tasks and model
(plan §2.3).

Grading needs ``refsol.py``, which is distributed separately. Without it this script
reports the format metrics (schema validity, completion, turns) and marks success rates
UNAVAILABLE. It never reports 0% for "could not grade".

    # no GPU, no server -- verifies the harness end to end
    python scripts/g2_model_gate.py --tasks data/test_data_v2.json --backend stub --limit 20

    # real run against `vllm serve`
    python scripts/g2_model_gate.py --tasks data/test_data_v2.json \\
        --backend vllm --base-url http://localhost:8000/v1 \\
        --fhir http://localhost:8080/fhir --refsol data/refsol.py \\
        --per-template 10 --concurrency 8 --out runs/g2_qwen3-8b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uqma.agent.loop import (  # noqa: E402
    STATUS_COMPLETED,
    STATUS_CONTEXT_LIMIT,
    STATUS_INVALID_ACTION,
    STATUS_LIMIT_REACHED,
    run_episode,
)
from uqma.agent.parsing import ActionKind  # noqa: E402
from uqma.envs.medagentbench import tasks as T  # noqa: E402
from uqma.envs.medagentbench.fhir import FhirClient, load_functions  # noqa: E402
from uqma.envs.medagentbench.grading import (  # noqa: E402
    GradingInput,
    RefsolGrader,
    grade_task,
)

ACTION_SR_THRESHOLD = 0.40
SCHEMA_VALID_THRESHOLD = 0.80


def select(tasks: list[T.Task], per_template: int | None, limit: int | None) -> list[T.Task]:
    """Stratified subsample: take the first N instances of every template.

    Stratifying by template rather than taking a head slice matters because the file is
    ordered by category -- a plain head would return only task1.
    """
    if per_template is not None:
        by_category: dict[str, list[T.Task]] = defaultdict(list)
        for task in tasks:
            by_category[task.category].append(task)
        chosen: list[T.Task] = []
        for category in sorted(by_category, key=lambda c: (len(c), c)):
            chosen.extend(by_category[category][:per_template])
        tasks = chosen
    return tasks[:limit] if limit else tasks


def build_backend(args) -> object:
    if args.backend == "stub":
        from uqma.inference.stub import CallableBackend

        return CallableBackend(_scripted_agent(args.fhir))

    from uqma.inference.openai_compat import OpenAICompatBackend

    backend = OpenAICompatBackend(
        base_url=args.base_url,
        model=args.model or "",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    if not backend.model:
        backend.discover_model()
    return backend


def _scripted_agent(api_base: str):
    """A deterministic fake agent for the no-GPU smoke path.

    Emits a plausible GET, then a POST for action templates, then FINISH -- enough to
    exercise every branch of the loop, the parser and the metrics without a model.
    """

    def respond(messages: list[dict]) -> str:
        opening = messages[0]["content"]
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")
        # Only the trailing "Question:" section, never the whole prompt -- the function
        # catalogue embedded above it contains words like "order" and would make every
        # task look like an action task.
        question = opening.rsplit("Question:", 1)[-1].lower()
        is_action = any(cue in question for cue in ("order", "record it", "referral"))

        if n_assistant == 0:
            return f"GET {api_base}/Observation?patient=S1234567&code=MG"
        if n_assistant == 1 and is_action:
            payload = {
                "resourceType": "Observation",
                "status": "final",
                "subject": {"reference": "Patient/S1234567"},
                "valueQuantity": {"value": 1.0, "unit": "mg/dL"},
            }
            return f"POST {api_base}/Observation\n{json.dumps(payload)}"
        return 'FINISH([1.0])'

    return respond


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-episode rows into the gate metrics.

    Kept separate from the run loop so it can be unit-tested on synthetic rows and
    re-run over a JSONL file without touching a GPU.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0}

    action_rows = [r for r in rows if r["is_action"]]
    query_rows = [r for r in rows if not r["is_action"]]
    graded = [r for r in rows if r["correct"] is not None]
    graded_action = [r for r in action_rows if r["correct"] is not None]
    graded_query = [r for r in query_rows if r["correct"] is not None]

    posts = sum(r["n_post_attempts"] for r in rows)
    valid_posts = sum(r["n_schema_valid_posts"] for r in rows)
    invalid_action_turns = sum(r["n_invalid_action_turns"] for r in rows)
    total_turns = sum(r["n_turns"] for r in rows)

    def rate(subset: list[dict]) -> float | None:
        return (sum(1 for r in subset if r["correct"]) / len(subset)) if subset else None

    tokens = sum(r["n_generated_tokens"] or 0 for r in rows)
    wall = sum(r["wall_s"] for r in rows)

    return {
        "n": n,
        "n_action": len(action_rows),
        "n_query": len(query_rows),
        "n_graded": len(graded),
        "overall_sr": rate(graded),
        "action_sr": rate(graded_action),
        "query_sr": rate(graded_query),
        "schema_valid_rate": (valid_posts / posts) if posts else None,
        "n_post_attempts": posts,
        "n_schema_valid_posts": valid_posts,
        "completion_rate": sum(1 for r in rows if r["status"] == STATUS_COMPLETED) / n,
        "invalid_action_rate": sum(1 for r in rows if r["status"] == STATUS_INVALID_ACTION) / n,
        "round_limit_rate": sum(1 for r in rows if r["status"] == STATUS_LIMIT_REACHED) / n,
        "context_limit_rate": sum(1 for r in rows if r["status"] == STATUS_CONTEXT_LIMIT) / n,
        "invalid_action_turn_rate": (invalid_action_turns / total_turns) if total_turns else None,
        "mean_turns": statistics.mean(r["n_turns"] for r in rows),
        "mean_wall_s": statistics.mean(r["wall_s"] for r in rows),
        "generated_tokens": tokens,
        "tokens_per_s": (tokens / wall) if wall > 0 else None,
        "per_template": _per_template(rows),
    }


def _per_template(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    for category, members in sorted(by_category.items(), key=lambda kv: _numeric(kv[0])):
        graded = [m for m in members if m["correct"] is not None]
        out[category] = {
            "n": len(members),
            "is_action": members[0]["is_action"],
            "sr": (sum(1 for m in graded if m["correct"]) / len(graded)) if graded else None,
            "completion_rate": sum(1 for m in members if m["status"] == STATUS_COMPLETED)
            / len(members),
            "mean_turns": statistics.mean(m["n_turns"] for m in members),
        }
    return out


def _numeric(category: str) -> tuple[int, str]:
    digits = "".join(c for c in category if c.isdigit())
    return (int(digits) if digits else 1 << 30, category)


def evaluate_gate(summary: dict, graded: bool) -> dict:
    if not graded:
        return {
            "gate": "INCOMPLETE",
            "reason": "refsol.py absent -- success rates unavailable, so G2 cannot be decided",
            "schema_valid_pass": (summary.get("schema_valid_rate") or 0) >= SCHEMA_VALID_THRESHOLD,
        }
    action_sr = summary.get("action_sr")
    schema = summary.get("schema_valid_rate")
    action_pass = action_sr is not None and action_sr >= ACTION_SR_THRESHOLD
    schema_pass = schema is not None and schema >= SCHEMA_VALID_THRESHOLD
    return {
        "gate": "PASS" if (action_pass and schema_pass) else "FAIL",
        "action_sr_pass": action_pass,
        "schema_valid_pass": schema_pass,
        "thresholds": {
            "action_sr": ACTION_SR_THRESHOLD,
            "schema_valid_rate": SCHEMA_VALID_THRESHOLD,
        },
    }


def render(summary: dict, gate: dict, label: str) -> str:
    def pct(value) -> str:
        return "  n/a " if value is None else f"{value * 100:6.2f}%"

    lines = [
        "=" * 78,
        f"GATE G2 -- MODEL GATE   [{gate['gate']}]   {label}",
        "=" * 78,
        f"tasks run            {summary['n']}  (action {summary['n_action']}, query {summary['n_query']})",
        f"graded               {summary['n_graded']}",
        "",
        f"action SR            {pct(summary['action_sr'])}   threshold >= 40%",
        f"query SR             {pct(summary['query_sr'])}",
        f"overall SR           {pct(summary['overall_sr'])}",
        f"schema-valid posts   {pct(summary['schema_valid_rate'])}   threshold >= 80%"
        f"   ({summary['n_schema_valid_posts']}/{summary['n_post_attempts']})",
        "",
        f"completion rate      {pct(summary['completion_rate'])}",
        f"invalid action       {pct(summary['invalid_action_rate'])}",
        f"round limit hit      {pct(summary['round_limit_rate'])}",
        f"context limit hit    {pct(summary['context_limit_rate'])}",
        f"mean turns           {summary['mean_turns']:.2f}",
        f"mean wall/task       {summary['mean_wall_s']:.2f}s",
        f"tokens/s             {summary['tokens_per_s']:.1f}" if summary["tokens_per_s"] else "tokens/s             n/a",
        "",
        "PER TEMPLATE",
    ]
    for category, stats in summary["per_template"].items():
        flag = "A" if stats["is_action"] else "q"
        lines.append(
            f"  {category:<10} [{flag}] n={stats['n']:<4} SR={pct(stats['sr'])} "
            f"complete={pct(stats['completion_rate'])} turns={stats['mean_turns']:.2f}"
        )

    if gate["gate"] == "INCOMPLETE":
        lines += ["", f"INCOMPLETE: {gate['reason']}"]
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate G2: model selection gate")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--functions", type=Path, default=Path("data/funcs_v1.json"))
    parser.add_argument("--fhir", default="http://localhost:8080/fhir")
    parser.add_argument("--refsol", type=Path, default=None)
    parser.add_argument("--backend", choices=("vllm", "stub"), default="vllm")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-round", type=int, default=5,
                        help="upstream default is 5; the proposal says 8 (Spike A resolves)")
    parser.add_argument("--per-template", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--skip-fhir-check", action="store_true")
    args = parser.parse_args(argv)

    tasks = select(T.load_tasks(args.tasks), args.per_template, args.limit)
    template_flags = {g.key: g.is_action for g in T.templates(T.load_tasks(args.tasks))}
    functions = load_functions(args.functions)

    fhir = FhirClient(args.fhir)
    if args.backend != "stub" and not args.skip_fhir_check and not fhir.verify():
        print(
            f"FHIR server not reachable at {args.fhir}. Start the Docker image first, or "
            "pass --skip-fhir-check to run anyway.",
            file=sys.stderr,
        )
        return 2

    grader = None
    if args.refsol:
        if args.refsol.exists():
            grader = RefsolGrader(args.refsol, args.fhir)
        else:
            print(f"warning: refsol not found at {args.refsol}; success rates will be "
                  "UNAVAILABLE", file=sys.stderr)

    backend = build_backend(args)
    # Graders take the raw dict, not our Task dataclass, so keep both keyed by id.
    raw = {item["id"]: item for item in _raw_tasks(args.tasks)}

    out_dir = args.out
    episodes_path = None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        episodes_path = out_dir / "episodes.jsonl"
        episodes_path.write_text("", encoding="utf-8")

    started = time.monotonic()
    rows: list[dict] = []

    def run_one(task: T.Task) -> tuple[dict, dict]:
        episode = run_episode(
            task, backend, fhir, functions, max_round=args.max_round, capture_logprobs=True
        )
        correct = grade_task(
            grader,
            raw[task.id],
            GradingInput(result=episode.result, history=episode.history, status=episode.status),
        )
        row = {
            "task_id": task.id,
            "category": task.category,
            "is_action": template_flags.get(task.category, False),
            "status": episode.status,
            "correct": correct,
            "n_turns": episode.n_turns,
            "n_post_attempts": episode.n_post_attempts,
            "n_schema_valid_posts": episode.n_schema_valid_posts,
            "n_invalid_action_turns": sum(
                1 for t in episode.turns if t.action_kind == ActionKind.INVALID.value
            ),
            "n_generated_tokens": sum(t.n_generated_tokens or 0 for t in episode.turns),
            "wall_s": episode.wall_s,
        }
        return row, episode.to_dict()

    # Episodes are streamed to disk and only the small metric rows are kept in memory --
    # full transcripts across a sweep do not fit comfortably in RAM.
    handle = episodes_path.open("a", encoding="utf-8") if episodes_path else None
    try:
        if args.concurrency > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = pool.map(run_one, tasks)
                for row, episode in results:
                    rows.append(row)
                    if handle:
                        handle.write(json.dumps(episode) + "\n")
        else:
            for index, task in enumerate(tasks, 1):
                row, episode = run_one(task)
                rows.append(row)
                if handle:
                    handle.write(json.dumps(episode) + "\n")
                print(f"  [{index}/{len(tasks)}] {task.id:<12} {row['status']:<22} "
                      f"turns={row['n_turns']} correct={row['correct']}", file=sys.stderr)
    finally:
        if handle:
            handle.close()

    summary = summarise(rows)
    summary["wall_s_total"] = time.monotonic() - started
    gate = evaluate_gate(summary, graded=grader is not None)

    label = f"{args.model or args.backend} | max_round={args.max_round}"
    print(render(summary, gate, label))

    if out_dir:
        (out_dir / "summary.json").write_text(
            json.dumps({"summary": summary, "gate": gate, "args": vars(args) | {
                "tasks": str(args.tasks), "functions": str(args.functions),
                "refsol": str(args.refsol) if args.refsol else None,
                "out": str(out_dir)}}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nwritten to {out_dir}/summary.json and {out_dir}/episodes.jsonl")

    return 0 if gate["gate"] == "PASS" else 1


def _raw_tasks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    raise SystemExit(main())
