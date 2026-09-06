# Runbook — bare GPU box to gate G2

Every command in order. Written for **Google Colab with an A100**, which is the clean
path: bfloat16 is supported and Qwen3-8B fits. Other targets are in the appendix.

For Colab specifically, `notebooks/uqma_colab.ipynb` is the same sequence as runnable
cells — open that instead of copying from here.

Gates: **G0** environment · **G1** task structure · **G2** model gate.

---

## 1. Repo and dependencies

```bash
git clone <repo> uqma && cd uqma
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest tests/ -q          # ~156 passed, 13 skipped
```

The package needs only `requests`; nothing heavy is imported outside the model server, so
every stage except serving runs on a laptop. The 13 skips are the grader tests.

## 2. The answer key

Download **https://stanfordmedicine.box.com/s/fizv0unyjgkb1r3a83rfn5p3dc673uho** and save
it as `data/refsol.py`. It is gitignored and **must never be committed** — MedAgentBench
distributes it out-of-band to keep it out of training crawls, and our derived turn-level
labels inherit that constraint.

Nothing is gradable without it: `sol` is null for 9 of the 10 templates, so the graders
recompute expected answers themselves.

```bash
.venv/bin/python -m pytest tests/ -q          # now ~169 passed
```

## 3. Gate G1 — task structure

No GPU, seconds. Run it before anything else: it bounds the statistical power available
to the whole thesis.

```bash
.venv/bin/python scripts/g1_task_structure.py --tasks data/test_data_v2.json \
    --refsol data/refsol.py --out results/g1_v2.json
```

## 4. FHIR server

Colab has no Docker, and neither do most managed GPU hosts. Irrelevant — the
MedAgentBench image is a Spring Boot HAPI FHIR war plus a preloaded H2 database, so a JVM
is enough. One idempotent command does everything:

```bash
bash scripts/bootstrap_fhir.sh ~/fhir
```

Installs Java if missing, pulls ~1.8 GB of image layers, verifies each against a manifest
pinned from Docker Hub, unpacks ~5 GB, starts the server detached, waits for it, and runs
G0's FHIR half. Re-running skips whatever is already done.

Expect `[PASS]` with **695 patients · 563,426 observations · 21,991 medication requests ·
124,969 procedures · 74,821 conditions**. Different numbers mean an incomplete extraction.

**On trust:** digest verification alone only proves a blob matches the manifest that named
it, and that manifest comes from the same registry as the blobs — so a hostile mirror
could serve a poisoned manifest plus matching poisoned layers and every check would pass.
`data/medagentbench_image_manifest.json` is pinned from Docker Hub and every resolved
manifest is compared against it before a byte downloads. Default is Docker Hub only;
`--try-mirrors` adds third-party pull-through mirrors for networks where Hub is blocked,
still pin-verified. Never pass `--allow-manifest-drift` to work around a mismatch you
cannot explain.

If Docker Hub is unreachable, `scripts/fetch_fhir_server.py --check --try-mirrors` reports
which registries serve both a manifest and a real blob, and prints the `--registry` line
to use.

## 5. Model server

```bash
.venv/bin/python scripts/gpu_profile.py
```

Prints the serve command for this GPU — correct dtype, largest model that fits, and the
flags the sweeps need. On an A100 that is Qwen3-8B in bfloat16:

```bash
pip install vllm

vllm serve Qwen/Qwen3-8B --served-model-name Qwen3-8B \
  --dtype bfloat16 --max-model-len 8192 \
  --enable-prefix-caching --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.90
```

`--enable-prefix-caching` because the k resampled actions at a turn share a long identical
prompt — without it the prefill is recomputed k times. `--no-enable-chunked-prefill`
because hidden-state extraction needs it later, and G2's throughput should be measured
under the configuration real sweeps use.

## 6. Gate G0

```bash
.venv/bin/python scripts/g0_environment.py \
    --fhir http://localhost:8080/fhir --base-url http://localhost:8000/v1 \
    --determinism-trials 8 --out results/g0.json
```

Checks the server is populated (not merely up), that logprobs come back, and measures
**logprob determinism**: vLLM is not bitwise deterministic across batch compositions, so
identical temperature-0 requests can drift, which would be a noise floor under every
estimator. Measured at exactly `0.000e+00` on an RTX 5090 — re-measure per GPU, and again
under concurrent load, since client-side repetition does not control batch composition.

## 7. Gate G2 — model gate

```bash
# harness check first: no GPU, ~1 min
.venv/bin/python scripts/g2_model_gate.py --tasks data/test_data_v2.json \
    --backend stub --per-template 1 --out runs/g2_smoke

# real, two scaffold tiers
.venv/bin/python scripts/g2_model_gate.py --tasks data/test_data_v2.json \
    --backend vllm --base-url http://localhost:8000/v1 \
    --fhir http://localhost:8080/fhir --refsol data/refsol.py \
    --per-template 5 --concurrency 2 --out runs/g2_nothink

.venv/bin/python scripts/g2_model_gate.py --tasks data/test_data_v2.json \
    --backend vllm --base-url http://localhost:8000/v1 \
    --fhir http://localhost:8080/fhir --refsol data/refsol.py \
    --per-template 5 --concurrency 2 --strip-think --out runs/g2_think
```

**Passes at action SR ≥ 40% and schema-valid ≥ 80%**, read off the *action* column — two
published open-weight models score 0.00% on action tasks while scoring 8–39% on query
tasks, and an all-negative label set makes AUROC undefined at exactly the turns the thesis
is about.

Both tiers are run because Qwen3 emits `<think>` blocks that our prefix dispatch
classifies as INVALID. `--strip-think` fixes that but is a **scaffold change**, not a
parity fix: scaffold quality alone moves success on this benchmark by more than 20 points,
so the delta is measured rather than folded in silently.

Then pick three models spanning action SR at roughly 25 / 45 / 65% and re-run the
shortlist on the full 300 (drop `--per-template`).

## 8. First sweep

```bash
.venv/bin/python scripts/run_agent.py --tasks data/test_data_v2.json \
    --backend vllm --base-url http://localhost:8000/v1 \
    --fhir http://localhost:8080/fhir --refsol data/refsol.py \
    --n-samples 10 --concurrency 8 --out runs/ --notes "first production sweep"
```

Immutable run directory with a manifest (git SHA, dirty flag, config hash, library
versions), `trajectories.jsonl` and `run_stats.json`. `--resume <run_dir>` skips tasks
already logged, so a sweep dying at task 250 does not cost 250 tasks of GPU time — which
matters on Colab, where sessions cap at ~12 hours.

Run a `--per-template 2` trial first and extrapolate from its `run_stats.json`: it is the
first real datapoint against the proposal's unvalidated 15.6 GPU-hours-per-sweep estimate.

---

## Appendix — other targets

### Colab, non-A100

`gpu_profile.py` will say so, but in short: a **T4 is compute capability 7.5 and has no
bfloat16**, and an 8B model in 16-bit does not fit its 16 GB. It caps out around
Qwen3-1.7B in float16, which will very likely fail G2 — that is the gate working. Note
also that a float16 run is **not directly comparable** to the bf16 runs the proposal
specifies (§6.6), because logits are the measurement instrument. An L4 gets bf16 and
about Qwen3-4B.

### Colab session limits

Sessions disconnect after ~90 minutes idle and cap at ~12 hours. Copy `results/` and
`runs/` to Drive before the runtime dies; `--resume` handles the rest.

### Hosts where Docker Hub is blocked

Use `--check --try-mirrors` to find a working registry. On AutoDL specifically, the
`network_turbo` accelerator is a proxy that must be **ON for GitHub and HuggingFace** and
**OFF for container registries** — they are mutually exclusive, and having it the wrong
way round produces confusing 403s. Also `export VLLM_USE_FLASHINFER_SAMPLER=0` on RTX 5090
(sm120), where FlashInfer's arch check fails, and `export HF_HUB_DISABLE_XET=1` where
HuggingFace's Xet backend bypasses mirrors and 401s.

### If no registry is reachable at all

Pull on any machine with Docker Hub access and copy `~/fhir/rootfs` plus `~/fhir/run.sh`
across. The tree is self-contained apart from the absolute paths in `run.sh`; fix those by
re-running the fetch with `--offline` on the target, or by editing `ROOTFS=` at its top.
