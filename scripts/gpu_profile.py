#!/usr/bin/env python3
"""Report the GPU and emit a serving command that matches it.

Exists because two silent traps cost more than they should:

* **bfloat16 needs compute capability 8.0.** A Colab T4 is sm75 and has none. vLLM either
  errors or quietly falls back, and a fallback is worse: proposal §6.6 mandates bf16
  because logits are the measurement instrument, so a dtype change is a change to the
  instrument and has to be recorded, not absorbed.
* **Weights must fit with room for the KV cache.** An 8B model in 16-bit is ~16 GB, which
  does not fit a 16 GB T4 at all, and is tight on a 24 GB L4.

Running this before ``vllm serve`` turns both into a printed recommendation.

    python scripts/gpu_profile.py                 # report and recommend
    python scripts/gpu_profile.py --json          # machine-readable
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys

# Bytes per parameter at 16-bit, plus the overhead vLLM needs beyond raw weights.
BYTES_PER_PARAM_16BIT = 2
ACTIVATION_OVERHEAD_GB = 2.5
# Enough KV cache for the multi-turn loop: 8k context, several concurrent episodes.
MIN_KV_CACHE_GB = 4.0

CANDIDATES = [
    ("Qwen/Qwen3-1.7B", 1.7),
    ("Qwen/Qwen3-4B", 4.0),
    ("Qwen/Qwen3-8B", 8.2),
    ("Qwen/Qwen3-14B", 14.8),
    ("Qwen/Qwen3-32B", 32.8),
]


def probe() -> dict:
    """Read GPU name, memory and compute capability from nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return {"available": False, "reason": f"nvidia-smi failed: {exc}"}
    if not out:
        return {"available": False, "reason": "nvidia-smi reported no GPUs"}

    gpus = []
    for line in out.splitlines():
        name, memory, cap = (part.strip() for part in line.split(","))
        major, _, minor = cap.partition(".")
        gpus.append({
            "name": name,
            "memory_gb": round(float(memory) / 1024, 1),
            "compute_capability": cap,
            "supports_bf16": int(major) >= 8,
        })
    return {"available": True, "gpus": gpus, "n_gpus": len(gpus)}


def recommend(gpu: dict) -> dict:
    """Pick dtype and the largest candidate model that leaves room for a KV cache."""
    dtype = "bfloat16" if gpu["supports_bf16"] else "float16"
    budget = gpu["memory_gb"] * 0.90 - ACTIVATION_OVERHEAD_GB - MIN_KV_CACHE_GB

    fits = [
        (name, params) for name, params in CANDIDATES
        if params * BYTES_PER_PARAM_16BIT <= budget
    ]
    model = fits[-1][0] if fits else None

    notes = []
    if not gpu["supports_bf16"]:
        notes.append(
            f"{gpu['name']} is compute capability {gpu['compute_capability']}; bfloat16 "
            "needs 8.0+. Serving in float16 instead -- RECORD THIS. Proposal §6.6 "
            "specifies bf16 because logits are the measurement instrument, so results "
            "from a float16 run are not directly comparable to a bf16 one."
        )
    if model is None:
        notes.append(
            f"No candidate fits {gpu['memory_gb']} GB with a usable KV cache. Options: a "
            "smaller model, a quantized checkpoint (but quantization perturbs the logits "
            "this thesis measures), or a larger GPU."
        )
    elif model != "Qwen/Qwen3-8B":
        notes.append(
            f"Qwen3-8B does not fit; recommending {model}. Gate G2 selects on action "
            "success rate, and smaller models are likelier to fail it -- that is G2 "
            "working, not a bug."
        )
    return {"dtype": dtype, "model": model, "budget_gb": round(budget, 1), "notes": notes}


def serve_command(model: str, dtype: str, memory_gb: float) -> str:
    max_len = 8192 if memory_gb >= 22 else 4096
    return (
        f"vllm serve {model} \\\n"
        f"  --served-model-name {model.split('/')[-1]} \\\n"
        f"  --dtype {dtype} --max-model-len {max_len} \\\n"
        f"  --enable-prefix-caching --no-enable-chunked-prefill \\\n"
        f"  --gpu-memory-utilization 0.90"
    )


def render(profile: dict) -> str:
    if not profile["available"]:
        return (f"no GPU: {profile['reason']}\n"
                "  Colab: Runtime -> Change runtime type -> GPU")

    lines = ["=" * 78, "GPU PROFILE", "=" * 78]
    for gpu in profile["gpus"]:
        bf16 = "yes" if gpu["supports_bf16"] else "NO (needs 8.0+)"
        lines.append(f"  {gpu['name']}  {gpu['memory_gb']} GB  "
                     f"sm{gpu['compute_capability']}  bfloat16: {bf16}")

    first = profile["gpus"][0]
    rec = recommend(first)
    lines += ["", f"dtype        {rec['dtype']}",
              f"model        {rec['model'] or 'NONE FITS'}",
              f"weight budget {rec['budget_gb']} GB (after activations and KV cache)"]
    if rec["notes"]:
        lines.append("")
        for note in rec["notes"]:
            lines.append(f"  ! {note}")
    if rec["model"]:
        lines += ["", "serve with:", "", serve_command(rec["model"], rec["dtype"],
                                                       first["memory_gb"])]
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report GPU and recommend serving config")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    profile = probe()
    if args.json:
        if profile["available"]:
            profile["recommendation"] = recommend(profile["gpus"][0])
        print(json.dumps(profile, indent=2))
    else:
        print(render(profile))
    return 0 if profile["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
