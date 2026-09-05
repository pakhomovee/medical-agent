# Week-1/2 Technical Spike Specification

**Project:** Turn-Level Uncertainty Quantification for Tool-Using Medical AI Agents
**Purpose:** Answer three questions that determine whether the thesis as scoped is viable, *before* any instrumentation code is written.
**Duration:** 2 weeks. **Hardware:** 1 GPU for Spike A, up to 4 for Spike C, none for Spike B.

Nothing in weeks 3+ starts until G0, G1 and G2 are recorded in §5 of this document.

The order matters. Spike B is CPU-only and is the cheapest of the three, but it is placed second because its interpretation depends on having the environment actually running. If Spike A stalls, run Spike B while debugging — it needs no GPU and no working server.

---

## Spike A — Environment and serving stack

**Question:** Does the whole stack run on this hardware, and does it reproduce a published number?

**Gate G0: reproduce a published MedAgentBench success rate for one open-weight model to within ±5 percentage points.**

Reproducing a published number is the only evidence that the harness, the FHIR server, the prompt template, the tool schema and the grader are all wired correctly. Every subsequent result depends on this and nothing else validates it.

### A.1 Environment bring-up

```bash
git clone https://github.com/stanfordmlgroup/MedAgentBench
cd MedAgentBench
cat README.md          # do this first — confirm exact image tag, ports and entrypoints
```

Then, following the README (the specifics below are the documented shape; confirm each against the README rather than trusting this file):

1. **FHIR server.** Pull and run the packaged HAPI FHIR Docker image; it serves on `localhost:8080`. Confirm it is populated:
   ```bash
   curl -s 'http://localhost:8080/fhir/Patient?_count=1' | head -c 500
   curl -s 'http://localhost:8080/fhir/Observation?_count=1' | head -c 500
   ```
   A live server with an empty dataset is the classic silent failure. Check a non-zero `total`.
2. **Reference solutions.** `refsol.py` is distributed separately (Box link in the README) and must be placed at `src/server/tasks/medagentbench/refsol.py`. Without it there is no grader.
3. **Harness.** Start the task workers, controller and assigner as documented (README describes ~20 workers on ports 5000–5015 with the controller on 5000). Results land in `outputs/MedAgentBenchv1/<model>/medagentbench-std/overall.json`.

### A.2 vLLM on sm120 (RTX 5090)

```bash
python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```
Confirm compute capability reports 12.0. Serve the candidate model in bf16:

```bash
vllm serve <MODEL> \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.90
```

Two flags are load-bearing:
- `--enable-prefix-caching` — the k resampled actions at a turn share a long identical prompt. Without it the prefill is recomputed k times and the whole sampling budget is wrong by roughly k×.
- `--no-enable-chunked-prefill` — **required** for hidden-state extraction. Set it now so that A.2 and A.4 timings are measured under the configuration the real sweeps will use.

### A.3 Quantization path (expected to fail; that is an acceptable outcome)

Attempt to load one FP8 and one NVFP4 checkpoint. Known open defects on sm120: block-scaled FP8 weight-loading failures, and NVFP4 checkpoints silently falling back to Marlin W4A16. Record the exact error text and vLLM version.

**bf16 is the plan of record for all primary experiments regardless of the outcome** — quantization perturbs the logits that are the measurement instrument. This test exists to scope RQ5 and to know now rather than in month 7.

### A.4 Hidden-state extraction smoke test

Confirm the March-2026 mechanism writes safetensors, and confirm its documented limitation directly:

```bash
# request with kv_transfer_params: {"hidden_states_path": "...", ...}, max_tokens=1
python - <<'PY'
from safetensors import safe_open
with safe_open("<path>.safetensors", framework="pt") as f:
    for k in f.keys():
        print(k, f.get_slice(k).get_shape())
PY
```

Expect `hidden_states` of shape `[num_tokens, num_extracted_layers, hidden_size]` covering **prompt tokens only**. Verify the token count equals the prompt length, not prompt + generated. This confirms the two-pass design in §8.4 of the proposal is necessary rather than merely cautious.

Record: bytes written for one action span at 4 layers; extrapolate to a full sweep and compare against the 0.4 GB/sweep estimate.

### A.5 Run the benchmark

Run one open-weight model with a published number (Qwen2.5-72B at 51.33% or Llama-3.3-70B at 46.33% are the reference points; if VRAM forces a smaller model, the gate becomes "runs cleanly and produces a plausible split", and G0 is deferred until a reference-size model can be run).

Record overall / query / action success rates separately. **Always record the split** — the aggregate hides the failure mode this project studies.

---

## Spike B — Task structure and grader inspection

**Question:** Are the 300 tasks parameterised templates over the patient pool, or 300 hand-written singletons?

**Gate G1: answered yes or no in writing, with evidence.**

This one question decides (a) whether the study has adequate statistical power, (b) whether MedAgentBench-Ambiguous is a week of work or two months, and therefore (c) whether RQ3 stays in the thesis. Resolving it in week 1 rather than at week 24 is the difference between a planned descope and an emergency one.

### B.1 What to read

```bash
find src/server/tasks/medagentbench -type f | head -50
python -c "import json;d=json.load(open('<tasks>.json'));print(len(d));print(json.dumps(d[0],indent=2))"
sed -n '1,200p' src/server/tasks/medagentbench/refsol.py
grep -n "def " src/server/tasks/medagentbench/refsol.py
```

### B.2 Questions each must answer

| # | Question | Where to look | What it decides |
|---|---|---|---|
| B1 | Do task instances share a template string with a substituted patient identifier? Count *distinct* prompt templates after masking identifiers. | task JSON | If ~10 templates × ~30 patients → re-instantiation viable. If ~300 distinct → not. |
| B2 | Is there one grader function per *category* or per *task*? | `grep -n "def " refsol.py` — count functions vs 300 | A per-category grader parameterised by patient is exactly what re-instantiation needs. |
| B3 | Are the graders parameterised by patient/expected value, or do they hard-code constants? | read 3–4 grader bodies in full | Hard-coded constants mean each new instance needs a new expected value derived from the FHIR server — still automatable, but more work. |
| B4 | Do POST checkers verify individual payload fields, or one aggregate boolean? | grader bodies | Determines how much work the "field-level graded labels" contribution actually is. |
| B5 | Do any gold trajectories or reference API-call sequences exist? | whole task dir | Almost certainly no. Confirm, because the intermediate-turn label design depends on it. |
| B6 | How are query answers compared — exact match, numeric tolerance, set membership? | query grader | Determines the GET-turn label definition. |

### B.3 Deliverable

A one-page note recording: number of distinct templates, number of grader functions, whether graders are patient-parameterised, and an estimated achievable N for action instances. **Send this to the supervisor.** It is the single most decision-relevant artifact of the two weeks.

---

## Spike C — Model selection gate

**Question:** Can any available open-weight model act competently enough that turn labels are informative?

**Gate G2: the primary model achieves ≥40% action success rate AND ≥80% schema-valid tool calls.**

### C.1 Why this gate exists

Published action success rates include Gemma2 at **0.00%** and Mistral v0.3 at **0.00%**, against 38.67% and 8.00% on query tasks respectively. A model at 0% action SR produces an all-negative label set at commitment turns — AUROC is undefined, and the entire thesis is about those turns.

The published open-weight results are 2024-era models and understate what current small tool-callers do. They must be re-measured, not assumed in either direction.

### C.2 Candidates

Qwen3-8B, Qwen3-14B, Qwen3-32B, gpt-oss-20b, Qwen3-30B-A3B (Q4), Llama-3.3-70B (W4A16).

### C.3 Measure, per model

Run the action split (or a stratified 50-task subsample if time-constrained — record which):

| Metric | Why |
|---|---|
| **Action SR** | The gate criterion. |
| **Query SR** | Reported separately; never aggregate the two. |
| **Schema-valid tool-call rate** | Fraction of emitted calls that parse and match the FHIR function schema. Distinguishes clinical error from format failure — a model failing on JSON syntax teaches nothing about uncertainty. |
| **Mean turns to completion** | Feeds the sweep cost model in §8.3. |
| **Round-limit exhaustion rate** | Tasks hitting the 8-round cap; a high rate signals a prompt/tooling problem, not model capability. |
| **Tokens/s and TTFT under the A.2 serving config** | Validates the GPU-hour budget on real numbers rather than estimates. |

### C.4 Selection rule

Choose three models spanning **action SR** at approximately 25% / 45% / 65%. Do not select on overall SR — the two are decoupled (Gemini-1.5 Pro: 52.67% query, 71.33% action).

**If no candidate clears G2:** stop and reconsider before writing instrumentation. The bottleneck is then the environment or the harness prompt, not the estimators, and no amount of uncertainty quantification fixes it. Escalate to the supervisor with the C.3 table.

---

## 4. What is explicitly *not* in scope for these two weeks

Estimator implementation, the re-instantiation harness, label design, and any analysis code. The temptation is to start instrumenting during week 1 because it is the interesting part. Resist it: G1 changes the label design and G2 changes which models the harness must support, so code written before both gates is likely to be rewritten.

---

## 5. Results record — fill in and send to supervisor at end of week 2

```
SPIKE A — ENVIRONMENT                                    date: __________
  FHIR server up, non-empty ...................... [ ] yes  [ ] no
  refsol.py in place ............................. [ ] yes  [ ] no
  vLLM version / torch / CUDA .................... ______________________
  compute capability reported .................... ______
  bf16 serving OK ................................ [ ] yes  [ ] no
  FP8 load ....................................... [ ] ok  [ ] failed: ________
  NVFP4 load ..................................... [ ] ok  [ ] fell back  [ ] failed
  hidden states: prompt tokens only confirmed .... [ ] yes  [ ] no
  bytes per action span (4 layers) ............... ______  → per sweep: ______
  model run ...................................... ______________________
  overall SR ____%   query SR ____%   action SR ____%
  published reference ____%  → delta ____pp
  G0 (within +/-5pp) ............................. [ ] PASS  [ ] FAIL

SPIKE B — TASK STRUCTURE                    ANSWERED 2026-09-05 (scripts/g1_task_structure.py)
  distinct prompt templates ...................... 10 of 300      [v2; v1 = 10 of 100]
  instances per template ......................... 30             [v1: 10]
  ACTION templates ............................... 5  (150 instances)
  distinct patients .............................. 98
  tasks carrying 'sol' ........................... 30/300  -> grading NEEDS refsol.py
  id-grouping vs mask-grouping ................... consistent
  v1 vs v2 ....................................... v1 STRICT SUBSET of v2 (98/100 identical)
                                                   -> use test_data_v2.json, the published 300
  G1 (templating confirmed) ...................... [X] YES

  STILL OPEN (needs the Box download of refsol.py):
  grader functions in refsol.py .................. ______  (expect task1..task10)
  POST checkers field-level ...................... [ ] field  [ ] aggregate boolean
  graders read the transcript (invertibility) .... [ ] yes  [ ] no
  gold trajectories exist ........................ [ ] yes  [ ] no   (expect no)
    -> rerun: python scripts/g1_task_structure.py --tasks data/test_data_v2.json \
                --refsol data/refsol.py --out results/g1_v2.json

SPIKE C — MODEL GATE                                     date: __________
  model              action SR   query SR   valid-call%   turns   tok/s
  ________________   ______     ______     ______        ____    ____
  ________________   ______     ______     ______        ____    ____
  ________________   ______     ______     ______        ____    ____
  ________________   ______     ______     ______        ____    ____
  primary model selected ......................... ______________________
  G2 (>=40% action SR, >=80% valid calls) ........ [ ] PASS  [ ] FAIL
  three study models (action SR ~25/45/65) ....... ______________________

DECISION
  Proceed to weeks 3-5 as planned ................ [ ] yes
  Scope changes required ......................... ______________________
```

---

## 6. Gate summary

| Gate | Criterion | If it fails |
|---|---|---|
| **G0** | Published SR reproduced ±5pp | Debug the harness. Nothing downstream is trustworthy until this passes. |
| **G1** | Tasks are patient-parameterised templates | **PASSED 2026-09-05.** 10 templates x 30 instances; 5 action templates. But power is bounded by *template* count (5), not instance count (150) -- see plan SS3 R2. Grader invertibility still open pending refsol.py. |
| **G2** | ≥40% action SR, ≥80% schema-valid calls | Stop before instrumentation. Escalate with the C.3 table — the bottleneck is the environment, not the estimators. |
