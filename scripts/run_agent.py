#!/usr/bin/env python3
"""Stage 1 -- run the agent and log trajectories.

The expensive GPU stage. It logs raw signal (generations, token logprobs, k resampled
actions) and computes no uncertainty scores: estimators are stage 4, off the artifact, so
that adding an estimator in month six costs a CPU pass rather than a re-sweep (plan §6.2).

    # smoke: all six stages' worth of plumbing, no GPU, no server, seconds
    python scripts/run_agent.py --tasks data/test_data_v2.json --backend stub \\
        --per-template 1 --out runs/

    # real sweep
    python scripts/run_agent.py --tasks data/test_data_v2.json \\
        --backend vllm --base-url http://localhost:8000/v1 \\
        --fhir http://localhost:8080/fhir --refsol data/refsol.py \\
        --n-samples 10 --concurrency 8 --out runs/

Resumable: pass ``--resume <run_dir>`` to skip tasks already present in that run's log.
A sweep that dies at task 250 should not cost 250 tasks of GPU time to restart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uqma.agent.loop import run_episode  # noqa: E402
from uqma.envs.medagentbench import tasks as T  # noqa: E402
from uqma.envs.medagentbench.fhir import FhirClient, load_functions  # noqa: E402
from uqma.envs.medagentbench.grading import (  # noqa: E402
    GradingInput,
    RefsolGrader,
    grade_task,
)
from uqma.trajectory.store import (  # noqa: E402
    Manifest,
    RunWriter,
    iter_trajectories,
)


def build_backend(args):
    if args.backend == "stub":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from g2_model_gate import _scripted_agent
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


def completed_task_ids(run_dir: Path) -> set[str]:
    if not (run_dir / "trajectories.jsonl").exists():
        return set()
    return {t.task_id for t in iter_trajectories(run_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1: run the agent, log trajectories")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--functions", type=Path, default=Path("data/funcs_v1.json"))
    parser.add_argument("--fhir", default="http://localhost:8080/fhir")
    parser.add_argument("--refsol", type=Path, default=None)
    parser.add_argument("--backend", choices=("vllm", "stub"), default="vllm")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="greedy action temperature; 0 for parity runs")
    parser.add_argument("--sample-temperature", type=float, default=1.0,
                        help="temperature for the k resampled alternatives")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="k alternatives per turn, history pinned (0 disables)")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-round", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-template", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--skip-fhir-check", action="store_true")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from g2_model_gate import select

    all_tasks = T.load_tasks(args.tasks)
    tasks = select(all_tasks, args.per_template, args.limit)
    functions = load_functions(args.functions)

    fhir = FhirClient(args.fhir)
    if args.backend != "stub" and not args.skip_fhir_check and not fhir.verify():
        print(f"FHIR server not reachable at {args.fhir}", file=sys.stderr)
        return 2

    grader = None
    if args.refsol and args.refsol.exists():
        grader = RefsolGrader(args.refsol, args.fhir)
    elif args.refsol:
        print(f"warning: refsol not found at {args.refsol}; trajectories will be ungraded",
              file=sys.stderr)

    backend = build_backend(args)

    config = {
        "tasks": str(args.tasks),
        "model": args.model or args.backend,
        "backend": args.backend,
        "temperature": args.temperature,
        "sample_temperature": args.sample_temperature,
        "n_samples": args.n_samples,
        "max_round": args.max_round,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "per_template": args.per_template,
        "limit": args.limit,
        "graded": grader is not None,
    }

    already: set[str] = set()
    if args.resume:
        already = completed_task_ids(args.resume)
        manifest = Manifest.create(config, run_id=args.resume.name, notes=args.notes)
        writer = RunWriter(args.resume.parent, manifest, overwrite=True)
        writer._handle = (args.resume / "trajectories.jsonl").open("a", encoding="utf-8")
        print(f"resuming {args.resume}: {len(already)} tasks already done", file=sys.stderr)
    else:
        manifest = Manifest.create(config, run_id=args.run_id, notes=args.notes)
        writer = RunWriter(args.out, manifest)

    if manifest.git_dirty:
        print("warning: working tree is dirty; this run is not reproducible from its SHA",
              file=sys.stderr)

    pending = [t for t in tasks if t.id not in already]
    raw = {item["id"]: item for item in json.loads(args.tasks.read_text(encoding="utf-8"))}

    def run_one(task):
        trajectory = run_episode(
            task, backend, fhir, functions,
            max_round=args.max_round,
            capture_logprobs=True,
            n_samples=args.n_samples,
            sample_temperature=args.sample_temperature,
            seed=args.seed,
            run_id=manifest.run_id,
        )
        trajectory.correct = grade_task(
            grader, raw[task.id],
            GradingInput.from_trajectory(trajectory),
        )
        return trajectory

    started = time.monotonic()
    done = 0
    try:
        if args.concurrency > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                for trajectory in pool.map(run_one, pending):
                    writer.write(trajectory)
                    done += 1
        else:
            for task in pending:
                trajectory = run_one(task)
                writer.write(trajectory)
                done += 1
                print(f"  [{done}/{len(pending)}] {task.id:<12} {trajectory.status:<22} "
                      f"turns={trajectory.n_turns} correct={trajectory.correct}",
                      file=sys.stderr)
    finally:
        writer.close()

    elapsed = time.monotonic() - started
    writer.write_artifact("run_stats.json", {
        "n_tasks": len(pending),
        "n_completed": done,
        "wall_s": round(elapsed, 2),
        "tasks_per_hour": round(done / elapsed * 3600, 1) if elapsed > 0 else None,
    })

    print(f"\n{done} trajectories -> {writer.dir}")
    print(f"wall {elapsed:.1f}s ({done / elapsed * 3600:.0f} tasks/h)" if elapsed > 0 else "")
    if grader is None:
        print("NOTE: ungraded (no refsol.py) -- trajectories carry correct=None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
