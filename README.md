# uqma — turn-level uncertainty quantification for tool-using medical AI agents

Instrumentation for the thesis described in
`Turn-Level Uncertainty Quantification for Tool-Using Medical AI Agents — Thesis Proposal.md`.
Execution plan: `~/.claude/plans/please-study-this-file-floating-dahl.md`.

Current status: **stage 1 built; gates G0, G1, G2 implemented.** G1 has been run and passes.

## Why this is not a fork of MedAgentBench

Upstream is an AgentBench fork with a controller/worker/assigner architecture that puts
an HTTP boundary between the harness and the model. We need per-token logprobs,
k-resampling with history pinned, and a teacher-forced second pass — none of which
survive that boundary cleanly. So upstream is pinned as a **data and grader source** and
the agent loop is ours, with `uqma/envs/medagentbench/prompts.py` transcribed verbatim so
gate G4 can check per-task parity.

## Install

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -e '.[dev]'          # add ,gpu on the GPU box
```

Data files (`test_data_v1.json`, `test_data_v2.json`, `funcs_v1.json`) come from
upstream's `data/medagentbench/` and are committed under `data/`.

**`refsol.py` is not committed and must never be.** It is the answer key, distributed via
a Box link in the upstream README specifically to keep it out of training crawls. Fetch it
to `data/refsol.py`, which is gitignored. Our derived turn-level labels inherit the same
constraint — see plan §6.1 on agreeing a gated release.

## Gates

### G1 — task structure (no GPU, no Docker, seconds)

```bash
.venv/bin/python scripts/g1_task_structure.py --tasks data/test_data_v2.json \
    --refsol data/refsol.py --out results/g1_v2.json
```

Answers whether tasks are parameterised templates (which bounds statistical power) and
whether graders can be inverted to a gold action (which the correct-and-continue deferral
design depends on). **Run this before anything else** — it has no dependencies and it is
the question that most changes the thesis.

Result on the shipped data, recorded in `results/g1_v2.json`:

| | v1 | v2 |
|---|---|---|
| tasks | 100 | **300** |
| templates | 10 | 10 |
| instances/template | 10 | 30 |
| action templates | **5** | **5** |
| action instances | 50 | 150 |
| tasks carrying `sol` | 10/100 | 30/300 |

`test_data_v1.json` is a strict subset of `test_data_v2.json` (same ids, 98/100 identical
content). **v2 is the published 300-task set — use it.** Note this is *not* the PSB 2026
paper's "300 new multi-step tasks", which live in a different repository despite the
identical filename.

Two consequences, both already folded into the plan:

- **Power is bounded by 5 action templates, not 150 instances.** Task-clustered bootstrap
  resamples templates; 30 MRN variations of one instruction are highly correlated. Adding
  instances does not add clusters.
- **`sol` is null for 9 of 10 templates**, so grading is impossible without `refsol.py`.
  G2 reports INCOMPLETE rather than 0% when it is absent.

### The FHIR server without Docker

Many GPU environments cannot run Docker at all — an AutoDL container drops
`cap_sys_admin`, so `dockerd` can never start regardless of privileges, and Docker Hub is
unreachable from some networks. Neither matters: the image is a Spring Boot HAPI FHIR war
plus a preloaded H2 database, so it runs on a bare JVM.

```bash
apt-get install -y openjdk-17-jre-headless

python scripts/fetch_fhir_server.py --out ~/fhir            # ~1.8 GB download
# behind a blocked Docker Hub:
python scripts/fetch_fhir_server.py --out ~/fhir --registry https://docker.m.daocloud.io

~/fhir/run.sh                                               # starts in ~70s
```

The puller resolves the manifest, downloads and digest-verifies each layer (cached and
resumable), applies whiteouts in order, and generates `run.sh` with the image's own
entrypoint — rewriting the absolute `/data` and `/configs` paths to the extracted tree so
no root-owned directories are needed.

**Budget ~6 GB of disk**: 1.8 GB compressed layers plus a 4.5 GB `test_db.mv.db`. Pass
`--keep-blobs` to retain the cache, `--heap 1200m` if RAM is tight.

Verified working: 695 patients, 563k observations, 125k procedures, 75k conditions.

### G0 — environment readiness (GPU box)

```bash
# either the extracted server above, or Docker if you have it
docker run -d -p 8080:8080 <hapi-fhir-image-from-upstream-readme>
vllm serve <MODEL> --dtype bfloat16 --max-model-len 8192 \
  --enable-prefix-caching --no-enable-chunked-prefill

.venv/bin/python scripts/g0_environment.py --fhir http://localhost:8080/fhir \
    --base-url http://localhost:8000/v1 --determinism-trials 8 --out results/g0.json
```

`--enable-prefix-caching` because k resampled actions at a turn share a long identical
prompt. `--no-enable-chunked-prefill` because hidden-state extraction requires it, and G2
throughput should be measured under the configuration real sweeps use.

G0 checks four things:

1. FHIR is up **and populated** — a live server with an empty dataset is the classic
   silent failure, where every GET succeeds, returns nothing, and the model looks
   incompetent.
2. The model server returns **per-token logprobs**. They are the measurement instrument
   for the whole thesis; finding them absent after a sweep is expensive.
3. The serving flags above.
4. **Logprob determinism.** vLLM is not bitwise deterministic across batch compositions,
   so the same prompt can yield slightly different logprobs run to run. That puts a noise
   floor under every estimator, and it can make the G4 parity check fail for reasons
   unrelated to our code. `--determinism-trials` issues N identical temperature-0
   requests and reports text divergence plus max/mean logprob drift. Client-side
   repetition cannot control batch composition, so treat the number as a **lower bound**.

### Stage 1 — sweeps

```bash
.venv/bin/python scripts/run_agent.py --tasks data/test_data_v2.json \
    --backend vllm --base-url http://localhost:8000/v1 \
    --fhir http://localhost:8080/fhir --refsol data/refsol.py \
    --n-samples 10 --concurrency 8 --out runs/
```

Writes a run directory named `<date>-<model>-<confighash>` holding `manifest.json` (git
SHA, dirty flag, resolved config, library versions), `trajectories.jsonl` and
`run_stats.json`. Runs are immutable; `--resume <run_dir>` skips tasks already logged, so
a sweep that dies at task 250 does not cost 250 tasks of GPU time to restart.

`--n-samples k` draws k alternative actions per turn with history pinned. They are
recorded, never executed, so sampling costs k× generation without multiplying environment
interactions.

### G2 — model gate (GPU box)

```bash
# harness check: no GPU, no server, no model
.venv/bin/python scripts/g2_model_gate.py --tasks data/test_data_v2.json \
    --backend stub --per-template 2 --out runs/g2_smoke

# real run
.venv/bin/python scripts/g2_model_gate.py --tasks data/test_data_v2.json \
    --backend vllm --base-url http://localhost:8000/v1 \
    --fhir http://localhost:8080/fhir --refsol data/refsol.py \
    --per-template 10 --concurrency 8 --out runs/g2_qwen3-8b
```

**Passes at action SR ≥ 40% and schema-valid tool-call rate ≥ 80%.**

Selection is on *action* success rate, never overall: two published open-weight models
score 0.00% on action tasks while scoring 8–39% on query tasks, and an all-negative label
set makes AUROC undefined at exactly the turns the thesis is about.

Start with `--per-template 10` (100 tasks) to rank candidates cheaply, then re-run the
shortlist on the full 300. Episodes stream to `episodes.jsonl` as they finish; only small
metric rows are held in memory.

## Layout

```
uqma/envs/medagentbench/   tasks (template analysis) · prompts (verbatim) · fhir · grading
uqma/agent/                parsing (+ canonicalisation) · loop · resample
uqma/trajectory/           schema (SCHEMA_VERSION 1.0.0) · store (manifests, JSONL)
uqma/inference/            base · openai_compat (vllm serve) · stub (no GPU)
scripts/                   g0_environment.py · g1_task_structure.py · g2_model_gate.py
                           run_agent.py (stage 1) · fetch_fhir_server.py
tests/                     117 tests, no GPU or network required
results/                   small gate reports (committed -- decision evidence)
runs/                      sweep artifacts (gitignored -- 1-2 TB)
```

The pipeline is six stages, each reading and writing durable artifacts so the expensive
GPU stage runs once and everything downstream is re-runnable on a laptop (plan §6.2).
Stage 1 is built; stages 2–6 are not.

### The trajectory schema

`uqma/trajectory/schema.py` is the load-bearing interface: stage 1 writes it, stages 2–6
read it and never touch a model. `SCHEMA_VERSION` is stamped into every manifest and
readers refuse a mismatch, so a change means writing a migration rather than teaching
readers two shapes.

Three deliberate choices:

- **Hidden states are referenced, never inlined.** A sweep produces 1–2 TB of them; the
  tabular log must stay small enough to load whole. `HiddenStateRef` is filled in by
  stage 2, which is a separate pass — so changing which layers you extract costs one
  cheap re-run instead of regenerating every trajectory.
- **Per-turn prompts are reconstructed, not stored.** `Trajectory.prompt_for_turn(i)`
  returns `history[:2i+1]`. Storing a copy per turn duplicates every prior FHIR response
  once per subsequent turn — O(n²) in turn count, and FHIR bundles are large.
- **Fields the plan anticipates exist now with defaults**: `prefix_source` for prefix
  seeding (§4.3), `gated` / `injected_action` for correct-and-continue deferral (§4.2),
  `error_class` for format-vs-clinical stratification (§4.4). Adding a field later means
  migrating a terabyte of logs; adding it now costs a default value.

Action canonicalisation (`uqma/agent/parsing.canonical_action`) normalises query-parameter
order, JSON key order and whitespace so that two samples can be compared exactly. That is
what makes semantic entropy cheap here — structured tool calls mean two generations agree
iff they are the same call with the same arguments, so no NLI model is needed.

## Upstream behaviours preserved on purpose

Gate G4 compares per-task outcomes against the official harness, so "fixing" any of these
registers as a parity failure:

- `'&_format=json'` is appended to GET URLs unconditionally, so a URL with no query string
  gets a stray `&`.
- A POST whose body is not JSON injects an error and the episode **continues**; only an
  unrecognised prefix terminates it.
- POSTs are never sent to the FHIR server — the payload is parsed for JSON validity and a
  canned success string is returned. `FhirClient(allow_writes=...)` exists only to make
  that decision visible; the plan says leave it off (§3 R4).
- `max_round` defaults to 5. The proposal text says 8 and costs at ~5 turns; Spike A
  resolves the discrepancy.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

98 tests. No GPU, no network, no Docker. 117 with `data/refsol.py` present, 104 + 13 skipped
without it, so a clean checkout still passes.

Of the three the plan calls load-bearing (§6.4): **grader mutation tests are written**
(`tests/test_grading.py` — a correct payload grades True, and seven mutations plus a
missing field and a double-POST all grade False); harness parity against upstream (gate
G4) and clustered-bootstrap coverage (stage 5) are not.

### Grading needs two shims

`refsol.py` opens with `from .utils import *` and calls `send_get_request`, so loading it
standalone needs a synthetic parent package — provided over our own `FhirClient`. And
`extract_posts` walks `results.history` expecting *objects* with `.role`/`.content` where
the assistant is spelled `'agent'` (AgentBench's convention), while our loop stores dicts
with `'assistant'` as the chat API requires. `GradingInput.from_trajectory` translates.
Always build grading input through it, never by hand.
