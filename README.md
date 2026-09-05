# uqma — turn-level uncertainty quantification for tool-using medical AI agents

Instrumentation for the thesis described in
`Turn-Level Uncertainty Quantification for Tool-Using Medical AI Agents — Thesis Proposal.md`.
Execution plan: `~/.claude/plans/please-study-this-file-floating-dahl.md`.

Current status: **gates G1 and G2 implemented.** G1 has been run and passes.

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
    --refsol data/refsol.py --out runs/g1_v2.json
```

Answers whether tasks are parameterised templates (which bounds statistical power) and
whether graders can be inverted to a gold action (which the correct-and-continue deferral
design depends on). **Run this before anything else** — it has no dependencies and it is
the question that most changes the thesis.

Result on the shipped data, already recorded in `runs/g1_v2.json`:

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

### G0 — environment and parity (GPU box)

```bash
docker run -d -p 8080:8080 <hapi-fhir-image-from-upstream-readme>
curl -s 'http://localhost:8080/fhir/Patient?_count=1' | head -c 300   # expect non-zero total

vllm serve <MODEL> --dtype bfloat16 --max-model-len 8192 \
  --enable-prefix-caching --no-enable-chunked-prefill
```

`--enable-prefix-caching` because k resampled actions at a turn share a long identical
prompt. `--no-enable-chunked-prefill` because hidden-state extraction requires it, and G2
throughput should be measured under the configuration real sweeps use.

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
uqma/agent/                parsing · loop
uqma/inference/            base · openai_compat (vllm serve) · stub (no GPU)
scripts/                   g1_task_structure.py · g2_model_gate.py
tests/                     55 tests, no GPU or network required
```

The pipeline is six stages, each reading and writing durable artifacts so the expensive
GPU stage runs once and everything downstream is re-runnable on a laptop (plan §6.2).
G1 and G2 are the front of stage 1; stages 2–6 are not built yet.

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

No GPU, no network, no Docker. The three that matter per plan §6.4 are grader mutation
tests (not yet written — they need `refsol.py`), harness parity against upstream (gate G4,
not yet written), and clustered-bootstrap coverage (stage 5, not yet built).
