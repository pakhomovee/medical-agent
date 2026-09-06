# Runbook — bare GPU box to gate G2

Every command in order. Written against an AutoDL container (no Docker, Docker Hub
blocked), but the AutoDL-specific parts are marked and skippable elsewhere.

Gates: **G0** environment · **G1** task structure · **G2** model gate.
G1 is already answered and committed (`results/g1_v2.json`); it needs no GPU.

---

## 0. AutoDL networking — read this first

AutoDL's accelerator is a proxy, and it is **mutually exclusive** with container
registries:

| Target | network_turbo |
|---|---|
| GitHub, HuggingFace | **ON** — `source /etc/network_turbo` |
| Docker registries and mirrors | **OFF** — `unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY` |

Most of the confusion in setup traces to having it the wrong way round. Keep two shells
if that is easier than remembering.

---

## 1. Repo and Python

```bash
source /etc/network_turbo                     # AutoDL: ON for GitHub
git clone <repo> ~/autodl-tmp/medical-agent
cd ~/autodl-tmp/medical-agent

python3 -m venv .venv --system-site-packages
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest tests/ -q          # expect ~131 passed, 13 skipped
```

The 13 skips are the grader tests; they need `refsol.py`, which comes next.

## 2. The answer key

Download **https://stanfordmedicine.box.com/s/fizv0unyjgkb1r3a83rfn5p3dc673uho** in a
browser and save it as `data/refsol.py`.

```bash
.venv/bin/python -m pytest tests/ -q          # now ~144 passed
.venv/bin/python scripts/g1_task_structure.py --tasks data/test_data_v2.json \
    --refsol data/refsol.py --out results/g1_v2.json
```

`refsol.py` is gitignored and **must never be committed** — it is the answer key,
distributed out-of-band to keep it out of training crawls.

## 3. FHIR server, no Docker

Docker is unavailable in an AutoDL container: `cap_sys_admin` is dropped from the
bounding set, so `dockerd` can never start regardless of user id. Irrelevant — the image
is a Spring Boot war plus an H2 database and runs on a JVM.

```bash
apt-get install -y openjdk-17-jre-headless
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY   # AutoDL: OFF

# which registries actually work from here? (~1 min)
.venv/bin/python scripts/fetch_fhir_server.py --check
```

It prints a ready-made `--registry` line. Use it:

```bash
.venv/bin/python scripts/fetch_fhir_server.py --out ~/autodl-tmp/fhir \
    --registry <first> --registry <second>
```

~1.8 GB down, ~5 GB unpacked (the H2 database is 4.5 GB of it). Layers are cached and
digest-verified, so an interrupted run resumes.

**On trusting mirrors.** Digest verification alone only proves a blob matches the manifest
that named it — and that manifest comes from the same registry as the blobs, so a hostile
mirror could serve a poisoned manifest plus matching poisoned layers and every check would
pass. `data/medagentbench_image_manifest.json` is pinned from Docker Hub and every
resolved manifest is compared against it; a mismatch aborts before a byte is downloaded.
That is what makes pulling through a third-party mirror safe. Refresh the pin only from
Docker Hub (`--pin-manifest`), and never pass `--allow-manifest-drift` to work around a
mismatch you have not explained.

Consider running the server as an unprivileged user: it is a 326 MB WAR from a
third-party image, and there is no reason for it to be root.

Then:

```bash
setsid nohup ~/autodl-tmp/fhir/run.sh > ~/autodl-tmp/fhir/server.log 2>&1 < /dev/null &

for i in $(seq 1 45); do
  curl -sf --max-time 3 http://localhost:8080/fhir/metadata >/dev/null && { echo UP; break; }
  sleep 5
done
grep -m1 "Started Application" ~/autodl-tmp/fhir/server.log
```

`setsid` matters: without it the server dies with the shell that launched it.

### If no registry is reachable

`--check` reporting nothing means egress filtering, not a bad mirror. Pull on any machine
with working Docker Hub access and copy `~/fhir/rootfs` and `~/fhir/run.sh` across — that
tree is self-contained apart from the absolute paths baked into `run.sh`, which you fix by
re-running the fetch with `--offline` on the target, or by editing `ROOTFS=` at its top.

## 4. Model server

```bash
source /etc/network_turbo                     # AutoDL: ON for HuggingFace
export HF_HOME=/root/autodl-tmp/hf            # ~16 GB; keep off the system disk
export HF_HUB_DISABLE_XET=1                   # Xet CAS bypasses mirrors and 401s
export HF_ENDPOINT=https://hf-mirror.com      # only if HF itself is slow

hf download Qwen/Qwen3-8B
```

```bash
export HF_HOME=/root/autodl-tmp/hf
export HF_HUB_OFFLINE=1                       # resolve from cache; fail fast, never hang
export VLLM_USE_FLASHINFER_SAMPLER=0          # FlashInfer's arch check breaks on sm120

vllm serve Qwen/Qwen3-8B --served-model-name Qwen3-8B \
  --dtype bfloat16 --max-model-len 8192 \
  --enable-prefix-caching --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.90
```

`--enable-prefix-caching` because k resampled actions at a turn share a long identical
prompt — without it the prefill is recomputed k times. `--no-enable-chunked-prefill`
because hidden-state extraction requires it later, and G2's throughput numbers should
match the config real sweeps use.

## 5. Gate G0

```bash
.venv/bin/python scripts/g0_environment.py \
    --fhir http://localhost:8080/fhir \
    --base-url http://localhost:8000/v1 \
    --determinism-trials 8 --out results/g0.json
```

Want `[PASS]` with:

```
Patient 695 · Observation 563426 · MedicationRequest 21991 · Procedure 124969 · Condition 74821
```

Different counts mean the extraction is incomplete. The determinism block measures
logprob drift across identical temperature-0 requests — it was exactly `0.000e+00` on an
RTX 5090, so there is no nondeterminism noise floor under the estimators. Re-measure if
the serving config changes.

## 6. Gate G2

```bash
# harness check: no GPU, ~1 min, catches config errors before spending GPU time
.venv/bin/python scripts/g2_model_gate.py --tasks data/test_data_v2.json \
    --backend stub --per-template 1 --out runs/g2_smoke

# real, two scaffold tiers
.venv/bin/python scripts/g2_model_gate.py --tasks data/test_data_v2.json \
    --backend vllm --base-url http://localhost:8000/v1 \
    --fhir http://localhost:8080/fhir --refsol data/refsol.py \
    --per-template 5 --concurrency 2 --out runs/g2_qwen3-8b_nothink

.venv/bin/python scripts/g2_model_gate.py --tasks data/test_data_v2.json \
    --backend vllm --base-url http://localhost:8000/v1 \
    --fhir http://localhost:8080/fhir --refsol data/refsol.py \
    --per-template 5 --concurrency 2 --strip-think --out runs/g2_qwen3-8b_think
```

**Passes at action SR ≥ 40% and schema-valid ≥ 80%**, read off the *action* column — two
published open-weight models score 0.00% on action tasks while scoring 8–39% on query
tasks, and an all-negative label set makes AUROC undefined at exactly the turns the thesis
is about.

Both tiers are run because Qwen3 emits `<think>` blocks by default, which our prefix
dispatch classifies as INVALID. `--strip-think` fixes that but is a **scaffold change**,
not a parity fix — scaffold quality alone moves success on this benchmark by more than 20
points, so the delta has to be measured rather than folded in silently.

Repeat per candidate model, restarting `vllm serve` between each. Then pick three spanning
action SR at roughly 25 / 45 / 65% and re-run the shortlist on the full 300 (drop
`--per-template`).

## 7. First sweep

```bash
.venv/bin/python scripts/run_agent.py --tasks data/test_data_v2.json \
    --backend vllm --base-url http://localhost:8000/v1 \
    --fhir http://localhost:8080/fhir --refsol data/refsol.py \
    --n-samples 10 --concurrency 8 --out runs/ --notes "first production sweep"
```

Immutable run directory with a manifest (git SHA, dirty flag, config hash, library
versions), `trajectories.jsonl` and `run_stats.json`. `--resume <run_dir>` skips tasks
already logged, so a sweep dying at task 250 does not cost 250 tasks of GPU time.

Check `run_stats.json` from a `--per-template 2` trial first and extrapolate — it is the
first real datapoint against the proposal's unvalidated 15.6 GPU-hours-per-sweep estimate.

---

## Known breakages and their fixes

| Symptom | Cause | Fix |
|---|---|---|
| `dockerd`: iptables permission denied | `cap_sys_admin` dropped in the container | Don't use Docker; §3 |
| `udocker`: `do not run as root` | udocker refuses uid 0 | `udocker --allow-root`, or an unprivileged user |
| Registry 403 that worked a minute ago | network_turbo proxy | `unset http_proxy …` |
| `git pull`: GnuTLS recv error | turbo is OFF and GitHub needs it | `source /etc/network_turbo` |
| Blob 404 from a mirror | pull-through cache never warmed | Resolve the manifest from the same registry — the script now always does |
| Download hangs on the first blob | urllib timeout is per read, not total | Fixed: stalls below 20 kB/s are abandoned |
| `hf download`: CAS 401 from xethub | Xet backend bypasses mirrors | `export HF_HUB_DISABLE_XET=1` |
| vLLM: FlashInfer requires sm75+ | arch detection fails on sm120 | `export VLLM_USE_FLASHINFER_SAMPLER=0` |
| vLLM: `Repo id must be in the form…` | a local path that does not exist | Serve by repo id with `HF_HOME` set |
| FHIR server dies when the shell exits | not detached | `setsid nohup … < /dev/null &` |
| G2 action SR ≈ 0, invalid-action ≈ 100% | model emits `<think>` | `--strip-think`, and report it as a scaffold tier |
