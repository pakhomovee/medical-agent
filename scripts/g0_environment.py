#!/usr/bin/env python3
"""Gate G0 -- environment readiness, plus a logprob determinism measurement.

Checks the three things a sweep silently depends on, and one thing nobody checks:

1. The FHIR server is up *and populated*. A live server with an empty dataset is the
   classic silent failure -- every GET succeeds, returns nothing, and the model looks
   incompetent.
2. The model server answers and returns per-token logprobs. Logprobs are the measurement
   instrument for the whole thesis; discovering they are absent after a sweep is
   expensive.
3. Serving flags. ``--enable-prefix-caching`` (k samples share a long prompt) and
   ``--no-enable-chunked-prefill`` (required for hidden-state extraction later).
4. **Determinism.** vLLM is not bitwise deterministic across batch compositions:
   continuous batching changes numerics, so the same prompt can yield slightly different
   logprobs run to run. That matters twice here -- it puts a noise floor under every
   estimator, and it can make gate G4's per-task parity check fail for reasons unrelated
   to our code. Nobody reports this number; measure it before trusting either.

    python scripts/g0_environment.py --fhir http://localhost:8080/fhir \\
        --base-url http://localhost:8000/v1 --determinism-trials 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uqma.envs.medagentbench.fhir import FhirClient  # noqa: E402

# Resource types the 300 tasks actually read. An empty count in any of these means the
# dataset did not load, whatever /metadata says.
EXPECTED_RESOURCES = ("Patient", "Observation", "MedicationRequest", "Procedure", "Condition")

DETERMINISM_PROMPT = [
    {"role": "user", "content": "Reply with exactly this and nothing else: GET /fhir/Patient?_id=S1"}
]


def check_fhir(api_base: str, timeout: float) -> dict:
    client = FhirClient(api_base, timeout=timeout)
    if not client.verify():
        return {"ok": False, "reachable": False, "error": f"no /metadata at {api_base}"}

    counts: dict[str, int | None] = {}
    for resource in EXPECTED_RESOURCES:
        result = client.get(f"{api_base}/{resource}?_summary=count&_format=json")
        if not result.ok or not isinstance(result.data, dict):
            counts[resource] = None
            continue
        counts[resource] = result.data.get("total")

    populated = all(isinstance(v, int) and v > 0 for v in counts.values())
    return {
        "ok": populated,
        "reachable": True,
        "counts": counts,
        "note": None if populated else "server is up but some resource types are empty",
    }


def check_model_server(base_url: str, timeout: float) -> dict:
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer EMPTY"})
    out: dict = {"ok": False, "base_url": base_url}

    try:
        response = session.get(f"{base_url}/models", timeout=timeout)
        response.raise_for_status()
        models = [m["id"] for m in response.json().get("data", [])]
    except (requests.RequestException, ValueError, KeyError) as exc:
        return out | {"error": f"cannot list models: {exc}"}

    if not models:
        return out | {"error": "server lists no models"}
    out["models"] = models
    model = models[0]

    try:
        response = session.post(
            f"{base_url}/chat/completions",
            json={"model": model, "messages": DETERMINISM_PROMPT, "temperature": 0.0,
                  "max_tokens": 32, "logprobs": True},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        return out | {"error": f"generation failed: {exc}"}

    choice = body["choices"][0]
    logprobs = (choice.get("logprobs") or {}).get("content")
    return out | {
        "ok": bool(logprobs),
        "model": model,
        "logprobs_available": bool(logprobs),
        "n_logprobs": len(logprobs) if logprobs else 0,
        "sample_text": (choice["message"]["content"] or "")[:120],
        "error": None if logprobs else
                 "server returned no logprobs -- every logit-based estimator is unavailable",
    }


def check_determinism(base_url: str, model: str, trials: int, timeout: float) -> dict:
    """Issue the same temperature-0 request repeatedly and measure drift.

    Client-side repetition cannot control batch composition directly, so this is a lower
    bound on the drift a real sweep sees: under concurrent load, batches vary more. Treat
    a clean result as "no worse than this", not as proof of determinism.
    """
    session = requests.Session()
    session.headers.update({"Authorization": "Bearer EMPTY"})

    texts: list[str] = []
    logprob_runs: list[list[float]] = []
    for _ in range(trials):
        try:
            response = session.post(
                f"{base_url}/chat/completions",
                json={"model": model, "messages": DETERMINISM_PROMPT, "temperature": 0.0,
                      "max_tokens": 32, "logprobs": True},
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
        choice = body["choices"][0]
        texts.append(choice["message"]["content"] or "")
        content = (choice.get("logprobs") or {}).get("content") or []
        logprob_runs.append([item["logprob"] for item in content])

    identical_text = len(set(texts)) == 1
    lengths = {len(run) for run in logprob_runs}
    max_delta = mean_delta = None
    if len(lengths) == 1 and logprob_runs and logprob_runs[0]:
        deltas = [
            max(abs(run[i] - logprob_runs[0][i]) for run in logprob_runs)
            for i in range(len(logprob_runs[0]))
        ]
        max_delta = max(deltas)
        mean_delta = statistics.mean(deltas)

    return {
        "ok": identical_text,
        "trials": trials,
        "identical_text": identical_text,
        "n_distinct_texts": len(set(texts)),
        "consistent_token_count": len(lengths) == 1,
        "max_logprob_delta": max_delta,
        "mean_logprob_delta": mean_delta,
        "note": (
            "Client-side repetition only; batch composition is not controlled, so real "
            "sweeps under concurrency may drift more. If max_logprob_delta is a material "
            "fraction of between-estimator differences, report it as a noise floor and "
            "pin batch settings for G4 parity runs."
        ),
    }


def render(report: dict) -> str:
    def mark(ok) -> str:
        return "PASS" if ok else "FAIL"

    fhir, model, det = report["fhir"], report["model_server"], report.get("determinism") or {}
    lines = [
        "=" * 78,
        f"GATE G0 -- ENVIRONMENT   [{report['gate']}]",
        "=" * 78,
        f"FHIR server          [{mark(fhir['ok'])}] {report['args']['fhir']}",
    ]
    if fhir.get("counts"):
        for resource, count in fhir["counts"].items():
            lines.append(f"    {resource:<20} {count if count is not None else 'ERROR'}")
    if fhir.get("note"):
        lines.append(f"    note: {fhir['note']}")
    if fhir.get("error"):
        lines.append(f"    error: {fhir['error']}")

    lines += ["", f"model server         [{mark(model['ok'])}] {model['base_url']}"]
    if model.get("model"):
        lines.append(f"    model              {model['model']}")
        lines.append(f"    logprobs           {model.get('n_logprobs', 0)} tokens")
        lines.append(f"    sample             {model.get('sample_text', '')!r}")
    if model.get("error"):
        lines.append(f"    error: {model['error']}")

    if det:
        lines += ["", f"determinism          [{'PASS' if det.get('ok') else 'WARN'}]  "
                      f"{det.get('trials', 0)} identical temperature-0 requests"]
        lines.append(f"    distinct outputs   {det.get('n_distinct_texts')}")
        if det.get("max_logprob_delta") is not None:
            lines.append(f"    max logprob drift  {det['max_logprob_delta']:.3e}")
            lines.append(f"    mean logprob drift {det['mean_logprob_delta']:.3e}")
        if det.get("error"):
            lines.append(f"    error: {det['error']}")
        lines.append(f"    {det.get('note', '')}")

    lines += [
        "",
        "REMINDERS",
        "  serve with: --dtype bfloat16 --enable-prefix-caching --no-enable-chunked-prefill",
        "  prefix caching: k samples share a long identical prompt (k x prefill without it)",
        "  chunked prefill off: required for hidden-state extraction (proposal §8.4)",
        "=" * 78,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate G0: environment readiness")
    parser.add_argument("--fhir", default="http://localhost:8080/fhir")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--determinism-trials", type=int, default=8,
                        help="identical temperature-0 requests; 0 to skip")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    fhir = check_fhir(args.fhir, args.timeout)
    model = (
        {"ok": False, "base_url": args.base_url, "error": "skipped"}
        if args.skip_model
        else check_model_server(args.base_url, args.timeout)
    )
    determinism = None
    if not args.skip_model and model.get("ok") and args.determinism_trials > 0:
        determinism = check_determinism(
            args.base_url, model["model"], args.determinism_trials, args.timeout
        )

    report = {
        "args": vars(args) | {"out": str(args.out) if args.out else None},
        "fhir": fhir,
        "model_server": model,
        "determinism": determinism,
        "gate": "PASS" if fhir["ok"] and (args.skip_model or model["ok"]) else "FAIL",
    }
    print(render(report))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nreport written to {args.out}")

    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
