# **Turn-Level Uncertainty Quantification for Tool-Using Medical AI Agents**

*Bachelor thesis proposal — Computer Science / Machine Learning*  
**Working title alternatives:** "Uncertainty quantification for medical AI agents" (broader, safer for registration) · "Knowing when to ask: uncertainty-aware deferral in EHR-acting agents" (sharper, better as a paper title).

*Revision 2. Changes from revision 1: scope is now explicitly tiered into committed / gated / appendix work (§4); the statistical protocol is specified rather than assumed (§6.6); model selection is gated on action-task success rather than overall success (§6.7); the research-gap table adds three missing neighbours (§3); several attributions are corrected. A summary of what changed and why is in §13.*

## ---

**1\. Project summary**

Large language model agents that read and write electronic health records currently succeed on roughly 50–70% of clinically derived tasks. At that level they cannot be deployed autonomously, and they cannot be deployed with a human either unless they can indicate *which* of their actions need review. This project asks whether an agent's own internal signals — token probabilities, sampling consistency, hidden-state geometry, stated confidence — can predict, at each individual step, whether the action it is about to take is wrong.  
The central hypothesis is that uncertainty should be measured and acted upon **asymmetrically**: an agent querying a lab value is performing a cheap, reversible read, whereas an agent submitting a medication order is making an irreversible commitment. A deferral policy that gates only commitments should dominate one that gates every step, both in error reduction and in the number of clinician interruptions it costs.

## **2\. Motivation**

Uncertainty estimation is the difference between a 70%-accurate agent being useless and being a safe assistive system. If the agent can flag the 30% of cases it is likely to get wrong, a clinician reviews those and the effective system accuracy approaches 100% at a bounded review cost. If it cannot, every output must be checked and the agent saves nobody any time.  
The failure mode to design against is alert fatigue: clinical decision support has a long history of alerts being overridden so routinely that they become noise. Consequently the primary evaluation metric of this project is a **cost curve — errors prevented per clinician interruption**.

This is an instance of **selective prediction** (Geifman & El-Yaniv 2017; Kamath et al. 2020) and of **learning to defer** (Madras et al. 2018; Mozannar & Sontag 2020), and it is reported here in the standard risk–coverage form so that it is comparable with that literature. What is *not* standard, and is the actual contribution, is the **asymmetric cost model**: coverage is not spent uniformly across turns but concentrated on the subset of turns that are irreversible. Framing the metric honestly as selective prediction under an asymmetric action cost is both more defensible and more useful than presenting it as a new measure.

## **3\. Research gap**

Eight adjacent bodies of work exist. None covers the intersection this project targets.

| Existing work | Covers | Does not cover |
| :---- | :---- | :---- |
| Mind the Gap (EACL 2026); When Confidence Fails (2026) | Medical LLM uncertainty and calibration; 1,810 questions across 11 specialties | Single-turn multiple-choice QA only; no tool use, no multi-step action |
| Knowing When to Abstain (EACL 2026) | Medical abstention under clinical uncertainty; conformal prediction \+ adversarial perturbation; explicit abstention options | Medical MCQA only; no agent, no tool call, no environment state |
| MedPRMBench (2026) | Step-level error detection in medical reasoning; 113,910 step-level labels | Reasoning chains only; explicitly excludes agents and tool use |
| ReDAct (2026); Dual-Process Agentic UQ (2026) | Uncertainty-aware deferral and control for LLM agents | ALFWorld, MiniGrid, WebShop, DeepResearch; no medical domain. **ReDAct defers to a larger model, not to a human** — it trades inference cost, not clinician attention, and therefore has no interruption budget |
| Argus (2026) | 27 UQ estimators across 7 families with rigorous metrics | Single-step GUI click grounding only; explicitly excludes multi-step reasoning; no medical domain |
| **Role-Stratified Conformal Risk Control for LLM Tool Calls (2026)** | **Per-field risk budgets over structured tool-call arguments; formal per-role guarantees. The nearest neighbour to this proposal's core idea** | **Security framing (prompt injection) on AgentDojo / InjecAgent; stratifies by *argument role* within a call, not by *reversibility* across turns; no medical domain; no clinician-interruption cost model.** CORA (2026) and ToolChain-CRC (2026) apply conformal risk control to GUI and tool-use agents on the same aggregate-call basis |
| FHIR-AgentBench (2025); FHIR-AgentEval (2026) | Realistic FHIR agent QA; 2,931 real clinical questions with ground-truth resource mapping | Question answering only — retrieval, not order entry; no irreversible write actions; no uncertainty or deferral component |
| Reinforced Hesitation; TIAR; Rewarding Intellectual Humility (2026) | RL training for abstention with tunable error-cost penalties | Single-turn QA only; no agents or tool use; no comparison against post-hoc uncertainty baselines |

Two published results establish that the gap is substantive rather than trivial:

> * The 2026 survey on uncertainty in LLM agents evaluated standard estimators on τ²-bench and reports **AUROC between 0.468 and 0.685** for predicting trajectory success (Table 2; the minimum is Kimi-K2.5 on Retail with token entropy, the maximum is GPT-4.1 on Telecom with verbalized confidence). The lower bound is worse than chance. Critically, that study used API-only models, so the logit-, hidden-state- and attention-based estimator families could not be evaluated at all.  
> * Argus found that estimator rankings transfer imperfectly **across datasets with the model held fixed** (mean Spearman ρ \= 0.705 over 120 dataset-shift pairs) and **essentially not at all across model tiers** (mean ρ ≈ 0.08 between open-weight and API-only models, 95% bootstrap CI \[−0.219, \+0.373\], which contains zero). Which estimator wins therefore cannot be looked up; it must be measured in the target setting.

Additionally, of 44 agent benchmarks surveyed, only **4 provide turn-level annotations**. This annotation scarcity is identified as the field's principal bottleneck.

**Position relative to the nearest neighbour.** Role-stratified conformal risk control establishes that treating a structured tool call as one undifferentiated unit hides risk in the fields that matter. This proposal makes the complementary argument one level up: treating a *trajectory* as one undifferentiated unit hides risk in the *turns* that matter, and the axis that matters clinically is reversibility. The two compose — and §6.5 adopts conformal risk control as the calibration layer for the commitment gate rather than competing with it.

## **4\. Research questions**

Scope is tiered. RQ1 and RQ2 are committed and constitute the thesis; RQ3 is a stretch gated on a Week-1 finding; RQ4 and RQ5 are appendix work and are the designated schedule buffer.

**Committed**

> 1. **RQ1 — Discrimination.** Can intrinsic uncertainty signals predict per-turn action correctness in a medical agent, and which estimator family performs best?  
> 2. **RQ2 — Aggregation.** Does gating irreversible commitments outperform uniform per-turn thresholding on the error-versus-interruption cost curve, **at matched interruption budget and against a position-only baseline**?

**Gated stretch** (proceeds only if the Week-1 task-templating finding in §6.3 holds)

> 3. **RQ3 — Abstention under missing information.** When a task is made genuinely unanswerable, do agents abstain, or do they act with unchanged confidence?

**Appendix / opportunistic**

> 4. **RQ4 — Transfer.** Do estimator rankings established on medical agents transfer to another agent environment, and do they change as base model capability increases? Primary transfer target is **FHIR-AgentBench** (same FHIR substrate, no user-simulator required, reuses the entire harness); τ²-bench is retained at trajectory level only, for direct comparison against the published 0.468–0.685 baseline.  
> 5. **RQ5 — Quantization.** Does weight quantization degrade uncertainty calibration even where it barely affects task accuracy?

## **5\. Datasets and environments**

### **Primary: MedAgentBench**

> * MIT licence; Docker-packaged HAPI FHIR server; publicly available. Also published in NEJM AI (10.1056/AIdbp2500144).  
> * 300 clinically derived tasks written by licensed physicians, over 100 synthetic patient profiles containing 700,000+ records.  
> * 10 task types across 7 categories: patient lookup, lab retrieval, data aggregation, recording vitals, test ordering, referral ordering, medication ordering with dose calculation.  
> * Nine FHIR API functions; maximum 8 interaction rounds per task; pass@1 grading.  
> * **150 query tasks (GET) and 150 action tasks (POST).** This split is the key methodological asset — see §6.2.  
> * Task prompts contain patient-identifier placeholders (`{MRN}`) and the grader and reference solution are curated **per task category**, not per task. Whether this makes the 300 tasks re-instantiable over the patient pool is the single most consequential open question about the design and is resolved in Week 1 (§6.3).

**Published success rates, split by task type.** The aggregate figure is misleading and revision 1 reported only the aggregate. Action success is neither monotone in overall success nor bounded below by it:

| Model | Overall | Query SR | Action SR |
| :---- | :---- | :---- | :---- |
| Claude 3.5 Sonnet v2 | 69.67% | 85.33% | 54.00% |
| GPT-4o | 64.00% | 72.00% | 56.00% |
| DeepSeek-V3 | 62.67% | 70.67% | 54.67% |
| Gemini-1.5 Pro | 62.00% | 52.67% | **71.33%** |
| GPT-4o-mini | 56.33% | 59.33% | 53.33% |
| o3-mini | 51.67% | 54.67% | 48.67% |
| Qwen2.5 | 51.33% | 38.67% | 64.00% |
| Llama 3.3 | 46.33% | 50.00% | 42.67% |
| Gemini 2.0 Flash | 38.33% | 34.00% | 42.67% |
| Gemma2 | 19.33% | 38.67% | **0.00%** |
| Gemini 2.0 Pro | 18.00% | 25.33% | 10.67% |
| Mistral v0.3 | 4.00% | 8.00% | **0.00%** |

The two 0.00% action rates drive the model-selection protocol in §6.7 and the revised risk table in §11.

### **Secondary: AgentClinic**

Open source. Doctor, patient, measurement and moderator agents over MedQA, MIMIC-IV, NEJM image cases; 7 languages, 9 specialties. Adds dialogue under incomplete information — the "should have asked another question" failure mode that MedAgentBench does not contain. Retained as context; not on the committed path.

### **Transfer: FHIR-AgentBench (primary), τ²-bench (secondary)**

FHIR-AgentBench grounds 2,931 real clinical questions in HL7 FHIR with ground-truth resource mapping. Because it shares the FHIR substrate, the instrumentation, estimators and label tooling built for MedAgentBench transfer directly, and it requires no user-simulator model. τ²-bench remains valuable as a non-medical control and as the only environment with a published agent-UQ baseline, but standing it up requires a second environment *and* a simulated user, so it is scoped to trajectory-level AUROC only.

### **Derived: MedAgentBench-Ambiguous (to be constructed, gated)**

A perturbation suite over MedAgentBench in which the correct action is made unavailable: the required lab value is deleted from the record, the patient identifier is made ambiguous, contradictory values are introduced, or a non-formulary medication is requested. This yields ground truth for "the correct behaviour is to ask, not to act", which does not currently exist for agents. Released as an artifact. Construction cost depends entirely on whether tasks are parameterised templates (cheap) or hand-written singletons (expensive) — hence the gate.

## **6\. Method and experimental design**

### **6.1 Unit of analysis**

> 1. **Trajectory** — label is task success; available directly from MedAgentBench.  
> 2. **Turn** — label is whether the individual action was correct and on-track.  
> 3. **Commitment** — uncertainty measured specifically at irreversible write actions.

**A caveat that must be stated rather than discovered at defence.** In MedAgentBench the irreversible write is typically the final action of an action task, so the correctness of the commitment turn is close to the trajectory label. Turn-level and trajectory-level analyses therefore partly collapse at exactly the turns of greatest interest. The scientifically meaningful framing is prospective: *can the agent's state before the write is emitted predict whether that write will be correct?* The intermediate turns, where no such collapse occurs, carry the genuinely novel signal.

### **6.2 Label construction**

The central methodological advantage of this environment: a FHIR write is a **structured object** with checkable fields — resource type, patient identifier, LOINC/NDC code, value, units, timestamp. Programmatic checkers therefore yield **automatic, per-turn correctness labels for the 150 action tasks without human annotation**. This directly addresses the turn-level annotation bottleneck.

**What already exists and what this project adds.** MedAgentBench ships `refsol.py`, which contains manually curated rule-based sanity checks over POST payloads. Binary end-of-task POST checking is therefore *not* a contribution of this project, and claiming it would be indefensible. The additions are:

> * **Field-level graded labels** rather than a single binary pass, so that a correct drug at the wrong dose is distinguishable from a wrong drug — which is what a reversibility-aware cost model needs.  
> * **Intermediate-turn labels** for the GET turns that precede a write, derived from a reference set of acceptable API calls per task category. These do not exist in any form today; MedAgentBench provides no gold trajectories.  
> * **A re-instantiation harness** (§6.3), if templating is confirmed.  
> * **A measured validity estimate.** A stratified subset of automatic labels is hand-audited and **Cohen's κ against the checker is reported**. An automatic label set that underpins every result in the thesis cannot go unaudited.

A second consequence of structured actions: semantic entropy, which normally requires a natural-language-inference model to decide whether two generations mean the same thing, becomes **exact** — two tool calls are equivalent if and only if they are the same call with the same arguments. This removes the largest source of noise in the sampling-based estimator family and is the single cleanest methodological advantage this environment offers.

### **6.3 Sample size and the re-instantiation question**

150 action tasks yield roughly 150–300 commitment turns and, at a ~40% error rate, only 60–120 error events. A 95% confidence interval on AUROC at that scale is approximately ±0.08–0.12 — wide enough that "estimator A beats estimator B" is not answerable, which would make RQ1 unfalsifiable as originally specified.

Because task prompts are placeholder-parameterised and graders are curated per category, the same graders should instantiate the same task templates against additional patients from the 100-patient pool at zero annotation cost, raising N to 500–1,000+ action instances. **Week 1 resolves this by direct inspection of `refsol.py` and the task definitions.** If confirmed, the re-instantiation harness is built in weeks 3–5 and RQ3 proceeds. If not, N stays at 150, RQ3 is dropped immediately rather than at week 24, and RQ1 is reported as a random-effects summary pooled across models with explicitly widened intervals.

Instantiations sharing a template are statistically correlated, so effective sample size grows sublinearly and all analysis clusters by template (§6.6).

### **6.4 Estimators compared**

Five committed families, following the Argus taxonomy where applicable, adapted to multi-turn:

> * **Logit-based:** mean token entropy, maximum sequence probability, perplexity over the action span.  
> * **Sampling / consistency:** self-consistency over k resampled actions; exact-match semantic entropy.  
> * **Density and probe:** Mahalanobis distance, SAPLMA-style linear probes on hidden states. Reported by Argus as the most *stable* family across regime changes, and therefore the highest-prior candidate.  
> * **Verbalized:** P(True), stated numeric confidence.  
> * **Learned:** a process-reward-style head trained on the automatic turn labels. *Note: Argus's seventh family is VLM-native (HEDGE-style perturbation), which has no text-agent analogue; substituting a learned head is a deliberate deviation from that taxonomy, not an instance of it.*

Promoted to stretch: **hybrid (CoCoA)** and **attention-based (RAUQ, Focus)**. RAUQ requires per-head predecessor-to-current-token attention weights. FlashAttention never materialises the attention matrix and vLLM exposes no API for it, so the attention family cannot run under the serving stack used for everything else. The fix is to compute attention-based scores inside the **second, teacher-forced pass** (§8.4) using HuggingFace transformers with `attn_implementation="eager"` and `output_attentions=True`, restricted to selected layers to bound O(L²) memory. That pass is already a single forward over a fixed sequence, so the marginal cost is small — but it is a distinct implementation and is therefore not on the committed path.

LM-Polygraph is used as a reference implementation and as a correctness cross-check on single-turn data.  
*Sampling protocol:* actions are resampled at turn t with history held fixed (cost ≈ k×), rather than resampling whole trajectories (combinatorial).

### **6.5 Aggregation — the proposed contribution**

Naive aggregation (mean, max, last-turn) is known to fail because an agent performing information-seeking actions legitimately exhibits high uncertainty while behaving correctly. The proposed alternative decomposes trajectory uncertainty into action uncertainty and observation uncertainty, and defines a **commitment risk**: the uncertainty present at the moment of an irreversible write. Reads are permitted freely; only writes are gated.

**The comparison must be constructed so that it can fail.** Because writes are usually terminal, gating only commitments issues fewer interruptions than uniform per-turn thresholding *by construction*, and a favourable cost curve could be an artifact of the benchmark's shape rather than evidence for the hypothesis. All comparisons are therefore made at **matched interruption budget** against four baselines:

> 1. Uniform per-turn thresholding.  
> 2. **Gate-last-turn-only with no uncertainty signal** — the critical ablation. It isolates whether uncertainty contributes anything beyond turn position.  
> 3. Random gating at matched budget.  
> 4. An oracle gate, as an upper bound.

If baseline 2 is not beaten, the finding is that *position*, not uncertainty, carries the effect, and it will be reported as such.

**Conformal risk control.** Split conformal risk control is applied to the commitment gate, converting the result from "here is a cost curve" into "post-gate error is bounded by α at interruption budget β, distribution-free, under exchangeability". This is calibration-set arithmetic with no GPU cost, it answers the alert-fatigue motivation in §2 in the form a clinical reader actually wants, and it positions the work explicitly alongside the role-stratified CRC line of work in §3 rather than in ignorance of it.

### **6.6 Metrics and statistical protocol**

> 1. **Cost curve:** errors prevented per clinician interruption, reported as a risk–coverage curve with AURC.  
> 2. **Conformal guarantee:** achieved risk at target α, and empirical coverage on a held-out fold.  
> 3. AUROC for turn correctness and trajectory failure.  
> 4. Prediction Rejection Ratio (PRR).  
> 5. Expected Calibration Error and Brier score.  
> 6. Cross-environment ranking transfer (Spearman ρ).

The statistical protocol is specified in advance because it determines whether any of the above can support a claim:

> * **The unit of resampling is the task template, not the seed.** Confidence intervals are task-clustered bootstrap intervals: resample templates with replacement, then instantiations within template. Reporting seed-variance intervals over a fixed task set is pseudo-replication — seeds resample the model, not the population the claim generalises over — and revision 1's "confidence intervals over seeds" is withdrawn.  
> * **A minimum detectable effect is fixed before the sweep runs.** Given N and the observed error rate, the AUROC difference detectable at 80% power is stated up front. Estimators that cannot be separated are reported as indistinguishable rather than ranked.  
> * **Results across three models are three correlated observations, not three replications.** A random-effects summary is used; no estimator is declared the winner on a 3-point ranking.  
> * **Multiple comparisons are corrected.** Roughly 5 families × 3 models × 2 label types ≈ 30 AUROCs; Benjamini–Hochberg is applied and reported.  
> * **Prevalence is always reported alongside AUROC**, which is prevalence-invariant while the cost curve is not.

### **6.7 Models and the selection gate**

Open-weight models only, so that token probabilities and hidden states are accessible. Primary experiments run in **bf16**, because quantization perturbs the logits that constitute the measurement instrument; quantized runs are confined to exploratory sweeps and to RQ5.

**Models are selected on action-task success rate, not overall success rate.** The two are decoupled (§5), and the failure mode this project studies lives in the action tasks. Revision 1 proposed "three models spanning the published success range (weak ≈ 20–40%)"; on the action split a weak model scores 0.00%, which yields all-negative labels and an undefined AUROC at precisely the commitment turns that are the subject of the thesis. That plan is withdrawn. A failure-rich regime is obtained instead from a *mid*-capability model and from MedAgentBench-Ambiguous.

The published open-weight results are 2024-era models. Modern small tool-callers are substantially better at structured function calling and must be measured rather than assumed. Candidates: Qwen3-8B / 14B / 32B, gpt-oss-20b, Qwen3-30B-A3B, Llama-3.3-70B (W4A16).

**Gate G2 (end of Week 2).** The primary model must achieve **≥40% action success rate** and **≥80% schema-valid tool calls**. If no candidate clears it, the environment — not the estimators — is the bottleneck, and the project stops to reconsider before any instrumentation is written. The three study models are then chosen to span action SR at roughly 25% / 45% / 65%.

## **7\. Work plan with hardware requirements**

All GPU figures assume 32 GB cards (RTX 5090 class). "Min VRAM" is the minimum per-GPU memory for that step; "GPUs" is the count needed concurrently. The FHIR server, metric computation, probe training and all analysis are CPU workloads requiring no GPU allocation.

### **Stage 0 — Minimum viable core (months 1–4)**

| Weeks | Work | GPUs | Min VRAM / GPU | Gate / notes |
| :---- | :---- | :---- | :---- | :---- |
| 1 | **Spike A.** MedAgentBench and Docker FHIR environment running; reproduce a published success rate for one open-weight model under vLLM. Validate the quantization path (see §8.4); disable chunked prefill; confirm hidden-state extraction writes safetensors. | **1** | **24 GB** | **G0: published SR reproduced within ±5pp.** Nothing else starts until this passes. |
| 1 | **Spike B.** Read `refsol.py` and the task definitions. Are tasks parameterised templates over patients? Are POST checkers field-level or binary? Do any gold trajectories exist? | **0** | — | **G1: decides §6.3 and RQ3.** CPU only. |
| 2 | **Spike C.** Measure action SR and schema-valid tool-call rate across candidate open-weight models. | **1–4** | **32 GB** | **G2: ≥40% action SR, ≥80% valid calls** (§6.7). |
| 3–5 | Instrument the agent loop: per-token log-probabilities, k resampled actions per turn, hidden-state extraction via the two-pass design. In parallel: re-instantiation harness if G1 passed. | **1** | **24 GB** | Second pass adds negligible VRAM; disk for safetensors. |
| 6–9 | Implement the five committed estimator families; cross-validate against LM-Polygraph on single-turn data. | **1** dev / **4** for batch runs | **24 GB** | Data-parallel replicas, not tensor parallel. |
| 10–12 | Build and validate automatic turn-level checkers; field-level graded POST labels; GET reference call sets; hand-audit a stratified subset and report κ. | **0–1** | **24 GB** | Predominantly CPU work. |
| 13–16 | First trajectory-level and turn-level AUROC across three models, with task-clustered intervals. | **4** | **24 GB** | **G3: is any estimator's lower CI bound above 0.5?** Decision point on annotation sufficiency. |

**Stage 0 deliverable:** a replication establishing whether the near-random agent-UQ result holds in a medical environment, using the logit-, hidden-state- and probe-access methods the prior τ²-bench study could not run at all. Publishable independently of everything that follows, and publishable whichever way it comes out.

### **Stage 1 — Thesis completion (months 5–9)**

| Weeks | Work | GPUs | Min VRAM / GPU | Notes |
| :---- | :---- | :---- | :---- | :---- |
| 17–20 | Full committed estimator sweep: 5 families × 3 models × multiple seeds, with multiple-comparison correction. | **4** | **32 GB** | 8B as 4 replicas (24 GB suffices); 14B needs TP=2 across 2×32 GB; 32B needs TP=4 across 4×32 GB. |
| 21–25 | Commitment-gated aggregation; cost curves at matched interruption budget versus all four baselines; conformal risk control on the gate (RQ2). | **0–1** | **24 GB** | Analysis on logged data; GPU only for targeted re-runs. **Do not compress — this is the contribution.** |
| 26–29 | Construct MedAgentBench-Ambiguous; measure abstention under missing information (RQ3). *Gated on G1.* | **4** | **24 GB** | 4 independent 8B replicas. |
| 30–32 | FHIR-AgentBench transfer; τ²-bench trajectory-level comparison; quantization–calibration study (RQ4, RQ5). | **4** | **32 GB** | **Designated schedule buffer — descope freely.** 70B W4A16 (~40 GB) requires TP=2 across 2×32 GB. |
| 33–38 | Thesis writing; artifact release and documentation. | **0** | — | No GPU allocation. |

**Buffer policy.** If Stage 0 overruns, cut RQ4/RQ5 first and RQ3 second. Weeks 21–25 and 33–38 are never compressed.

**Stage 1 deliverables:** a turn-level annotated medical agent benchmark; a commitment-gated deferral policy with cost curves and a conformal guarantee; the ambiguity perturbation suite; released code.

### **Stage 2 — Continuation beyond the diploma**

The agent-UQ survey names four open problems — benchmarks at scale, long-horizon black-box estimation, solution multiplicity, and multi-agent or self-evolving systems. This project addresses the first and engages the third. Remaining directions, each of master's or doctoral scale:

> * **Reinforcement learning for native abstention and clarification** — the most direct continuation. See §9 for why it belongs here rather than in the diploma, and what this project must deliver first to make it possible.  
> * Multi-agent clinical systems, where uncertainty must propagate across a consulting team and debate collapse is a failure mode.  
> * Non-stationary uncertainty in agents that adapt across episodes.  
> * Formal belief representations for unreliable tool outputs — in an EHR, stale laboratory values and unit inconsistencies.  
> * Validation on a real EHR with a clinical partner.

## **8\. Compute and infrastructure**

### **8.1 Hardware configuration rationale**

The workload is inference, not training, and is embarrassingly parallel across independent tasks. Consumer Blackwell GPUs have no NVLink, and measured tensor-parallel speedup on such cards is only 1.14–1.57×. The chosen configuration is therefore **four independent single-GPU model replicas (data parallelism)** rather than one tensor-parallel instance, giving near-linear scaling. Tensor parallelism is used only where a model does not fit in one card.

### **8.2 Model to VRAM mapping**

| Model | Precision | Weights | Min VRAM | GPUs per replica | Role |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 8B | bf16 | 16 GB | 24 GB | 1 | Primary — full logit fidelity, fast iteration |
| 14B | bf16 | 28 GB | 32 GB (tight; needs FP8 KV cache) or 2×32 GB | 1–2 (TP=2) | Mid-capability comparison |
| 32B | bf16 | 64 GB | 4×32 GB | 4 (TP=4) | Dense large-model chapter |
| 30–35B-A3B (MoE) | Q4 | ~20 GB | 32 GB | 1 | Best tool-use quality per GB; quantized, so RQ5 only |
| 70B | W4A16 | ~40 GB | 2×32 GB | 2 (TP=2) | Large-model comparison |

Final selection is subject to gate G2 (§6.7).

### **8.3 Per-sweep and total budget**

One full sweep \= 300 tasks × ~5 turns × 10 samples per turn. If re-instantiation (§6.3) succeeds, per-sweep cost scales with the instance count and the sweep counts below are reduced accordingly to hold the total constant.

| Configuration | GPUs | Min VRAM | Wall-clock | GPU-hours per sweep |
| :---- | :---- | :---- | :---- | :---- |
| 8B, bf16, 4 replicas (primary) | 4 | 24 GB | 3.9 h | 15.6 |
| 14B, bf16, TP=2, 2 replicas | 4 | 32 GB | 7.1 h | 28.4 |
| 32B, bf16, TP=4, 1 replica | 4 | 32 GB | 12.0 h | 48.1 |
| 70B, W4A16, TP=2, 2 replicas | 4 | 32 GB | 17.4 h | 69.4 |

| Budget item | GPUs held | GPU-hours |
| :---- | :---- | :---- |
| Stage 0 — interactive development | 1 | 480 |
| Stage 0 — exploratory sweeps (25 × 8B) | 4 | 391 |
| Stage 1 — interactive development | 1 | 320 |
| Stage 1 — production sweeps (40 × 8B, 15 × 14B, 8 × 32B, 3 × 70B) | 4 | 1,644 |
| Contingency (~25%) | — | 709 |
| **Total** | — | **≈ 3,550 GPU-hours** |

Interactive-development hours are doubled relative to revision 1. 240 GPU-hours across the sixteen weeks that contain all instrumentation debugging is 15 held-GPU-hours per week, which is not a realistic figure for the phase in which the harness is being built. Since the hardware is dedicated for the full period, the correction costs nothing.

Peak concurrent requirement is **4 GPUs × 32 GB**. The majority of calendar time requires only **1 GPU × 24 GB**. Storage: approximately 1–2 TB for trajectory logs, sampled actions and hidden states.

### **8.4 Software and known issues**

> * **vLLM** for serving, with automatic prefix caching enabled — the k samples at a given turn share a long identical prompt, and without prefix caching the prefill is recomputed k times.  
> * **Hidden-state extraction** is officially supported in vLLM as of March 2026 but **saves prompt tokens only** ("Only the prompt tokens and their hidden states will be saved"; `max_tokens=1` is the documented recommendation). A two-pass design is therefore required: generate normally, then re-submit the completed action as a prompt to capture its hidden states. Storage, sliced to the final token positions before each action, is approximately 0.4 GB per sweep for four layers of an 8B model.  
> * **Chunked prefill must be disabled** for hidden-state extraction. This is not optional and it changes long-context throughput; the sweep timings in §8.3 assume it is off.  
> * **Attention weights are not obtainable from vLLM at all.** FlashAttention does not materialise the attention matrix and no API exposes it. Attention-based estimators must run in HuggingFace transformers with eager attention, folded into the second pass (§6.4). This is the reason that family is not on the committed path.  
> * **Known risk:** FP8 and NVFP4 quantization paths have open defects on sm120 (RTX 5090) in vLLM — block-scaled FP8 weight loading failures and NVFP4 checkpoints falling back to Marlin W4A16. The quantization path must be validated in week 1; bf16 is the fallback and is in any case preferred for the primary experiments.

## **9\. Why reinforcement learning for abstention is Stage 2, not Stage 1**

Training an agent to abstain natively — rather than measuring its uncertainty post hoc — is the most compelling continuation of this work. It is deliberately excluded from the diploma for three reasons.

### **9.1 It requires this project's output as an input**

RL abstention methods use a ternary reward: \+1 for a correct action, −λ for an incorrect one, and a tunable value for abstaining. In the agentic setting that reward must be computed **per turn**. The field-level graded turn checkers built in Stage 0 (§6.2) are precisely that reward function. Without them there is no reward signal, and manual per-turn annotation at RL scale is infeasible.

### **9.2 The post-hoc baseline does not currently exist**

The published RL abstention literature — Reinforced Hesitation, TIAR, Rewarding Intellectual Humility — does not compare against post-hoc uncertainty baselines. Without knowing what a frozen model's own signals already achieve, an RL improvement cannot be attributed to learned abstention rather than to additional training in general. This project supplies that missing baseline.

### **9.3 Compute**

Published GRPO+LoRA abstention runs take approximately 24 hours per run on datacentre GPUs for *single-turn* QA. Multi-turn agentic rollouts are roughly 4× more expensive, and the central experiment is a sweep over the error-penalty λ to trace the Pareto frontier.

| Item | GPUs | Min VRAM | GPU-hours |
| :---- | :---- | :---- | :---- |
| One agentic GRPO+LoRA run, 8B | 4 (2 generation \+ 2 training, or FSDP+colocated vLLM) | 32 GB | ≈ 480 |
| λ sweep over 3 values \+ 3 evaluation seeds each | 4 | 32 GB | ≈ 1,580 |
| λ sweep over 5 values \+ 3 evaluation seeds each | 4 | 32 GB | ≈ 2,630 |

A minimal credible RL study therefore costs roughly half of this entire diploma's compute budget, and a full one costs nearly all of it. It is a separate project, not a chapter.

### **9.4 What it would add that post-hoc methods cannot**

> * **Post-hoc estimation is bounded by signal that already exists.** If a model's internals do not encode that it is about to err, no estimator can extract it. RL can create that signal by changing the policy.  
> * **Post-hoc deferral can only block; RL can substitute a better action.** A threshold says "stop, call a clinician." A trained policy can instead query the record for the missing value, then proceed — which is what a competent clinician does.  
> * **Prompting demonstrably cannot achieve this.** Reinforced Hesitation reports that frontier models almost never abstain on GSM8K, MedQA and GPQA even when explicitly warned of severe penalties, concluding that prompts cannot override training that rewards any answer over no answer. Behaviour change requires training, not instruction.

### **9.5 Known risks of the RL approach**

> * **Over-abstention collapse.** Reported abstention rates reaching 100% at aggressive penalties, with models unable to recover during subsequent RL. The mechanism has since been formalised — abstention as an action can null both the reward gradient and the KL anchor (2026) — along with proposed repairs.  
> * **Insufficient exploration.** Models initially emit "I don't know" too rarely for RL to obtain learning signal; supervised warm-start is typically required.  
> * **RLHF is known to worsen calibration.** Reward models systematically prefer higher-confidence responses regardless of correctness, so naive RL can increase overconfidence.  
> * **Fixed ternary rewards imply a fixed abstention threshold** independent of task difficulty; difficulty-aware advantage reweighting (TIAR) is an active research problem.

## **10\. Deliverables**

> 1. A turn-level annotated medical agent benchmark derived from MedAgentBench, with field-level graded structured-action checkers and intermediate-turn labels — which doubles as the reward function for future RL work. Reported with a measured label-validity κ.  
> 2. MedAgentBench-Ambiguous: a perturbation suite providing ground truth for justified abstention. *(Gated on G1.)*  
> 3. A comparative evaluation of five uncertainty estimator families on multi-step medical agents, with cost curves and pre-specified statistics.  
> 4. A commitment-gated deferral policy with a conformal risk guarantee, evaluated at matched interruption budget against a position-only ablation.  
> 5. Open-source instrumentation code for uncertainty logging in agent loops.  
> 6. The written thesis, and a workshop or conference submission.

## **11\. Risks and mitigations**

| Risk | Mitigation |
| :---- | :---- |
| **Degenerate action-task labels.** A weak open-weight model scores 0.00% on MedAgentBench action tasks, producing all-negative labels and an undefined AUROC at commitment turns. | Gate G2 (§6.7): select on *action* SR, require ≥40% action SR and ≥80% schema-valid calls for the primary model. Obtain the failure-rich regime from a mid-capability model and from MedAgentBench-Ambiguous, never from a weak model. Revision 1's "include a deliberately weak model" mitigation is withdrawn — it caused this risk rather than mitigating it. |
| **Insufficient statistical power.** 150 action tasks give ±0.08–0.12 intervals on AUROC, too wide to rank estimators. | Re-instantiate task templates over the patient pool (§6.3) to reach 500–1,000+ instances. Task-clustered bootstrap, pre-registered minimum detectable effect, random-effects pooling across models, Benjamini–Hochberg (§6.6). If templating fails, report indistinguishability honestly rather than a spurious ranking. |
| **RQ2 trivially true.** Writes are terminal, so gating writes issues fewer interruptions by construction. | Matched interruption budget; mandatory gate-last-turn-only ablation; oracle upper bound (§6.5). Report position-not-uncertainty if that is the answer. |
| **Solution multiplicity.** Multiple valid action sequences exist per task, so high uncertainty may reflect legitimate ambiguity rather than error. | Concentrate on irreversible actions with machine-checkable fields, where correctness is well defined. Report the ambiguity rate explicitly. |
| **Label validity.** Automatic checkers underpin every result. | Stratified hand-audit with reported Cohen's κ (§6.2). Treat "on-track" intermediate labels as explicitly weaker than commitment labels. |
| **Sparse failures.** At ~50% task success over ~5 turns, incorrect turns are a minority of all turns. | Report prevalence alongside AUROC; report intervals rather than point estimates. |
| **Instrumentation overrun.** Threading logprobs and hidden states through the agent harness. | Budgeted 3 weeks in Stage 0; vLLM's official hidden-state support materially de-risks this. Attention-based estimators are off the committed path precisely because they need a separate stack. |
| **Being scooped.** The field moves quickly, and role-stratified conformal risk control (2026) is close to the aggregation idea. | The primary deliverable is a benchmark and annotation resource, which retains value irrespective of who publishes which method. The differentiators — reversibility axis, clinical interruption cost, medical environment — are stated explicitly in §3. |
| **Null result.** Estimators may remain near-random. | A null result mirrors the established medical-QA abstention finding and is itself publishable; the benchmark and perturbation suite are delivered regardless. |
| **Simulated environment.** MedAgentBench uses synthetic patients; no clinical validation is possible. | Stated explicitly in the proposal and thesis. Clinical validation is scoped as Stage 2 work requiring a partner. No IRB is required for synthetic records. |
| **Disciplinary fit.** Agent/LLM research with a medical application rather than learning from biological data. | To be confirmed with the supervisor before registration. |

## **12\. Key references**

**Agent uncertainty and deferral**

> * Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities (ACL 2026). arXiv:2602.05073  
> * Argus / Uncertainty Quantification for Computer-Use Agents (ECCV 2026). arXiv:2606.25760  
> * ReDAct: Uncertainty-Aware Deferral for LLM Agents (2026). arXiv:2604.07036  
> * Agentic Uncertainty Quantification / Dual-Process AUQ (2026). arXiv:2601.15703  
> * Beyond Aggregate Risk: Role-Stratified Conformal Risk Control for LLM Tool Calls (2026). arXiv:2607.24343  
> * CORA: Conformal Risk-Controlled Agents for Safeguarded Mobile GUI Automation (2026). arXiv:2604.09155  
> * ToolChain-CRC: Conformal Risk Control for Agentic AI Under Retrieval and Tool-Use Drift (2026). arXiv:2606.18467  
> * Structured Uncertainty Guided Clarification for LLM Agents (SAGE-Agent). Adobe Research

**Environments**

> * MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents. arXiv:2501.14654; NEJM AI 10.1056/AIdbp2500144; github.com/stanfordmlgroup/MedAgentBench  
> * FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering. arXiv:2509.19319  
> * FHIR-AgentEval: A Modular Sandbox for Benchmarking Clinical LLM Agents. PMC12919212  
> * AgentClinic: a multimodal benchmark for tool-using clinical AI agents. agentclinic.github.io  
> * τ²-bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982

**Medical uncertainty, calibration and abstention**

> * Mind the Gap: Benchmarking LLM Uncertainty and Calibration with Specialty-Aware Clinical QA. EACL 2026; arXiv:2506.10769  
> * Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty. EACL 2026; arXiv:2601.12471  
> * When Confidence Fails: Overconfidence in LLMs under Uncertainty and Missing Clinical Information (2026). arXiv:2608.09080  
> * MedPRMBench: A Fine-grained Benchmark for Process Reward Models in Medical Reasoning (2026). arXiv:2604.17282

**Abstention training**

> * Honesty over Accuracy: Trustworthy Language Models through Reinforced Hesitation. arXiv:2511.11500  
> * TIAR: Trajectory-Informed Advantage Reweighting for LLM Abstention Learning (2026). arXiv:2605.25850  
> * Rewarding Intellectual Humility: Learning When Not to Answer in Large Language Models (2026). arXiv:2601.20126  
> * Abstention as an Action Can Kill Both the Reward Gradient and the KL Anchor (2026). arXiv:2608.00301  
> * Taming Overconfidence in LLMs: Reward Calibration in RLHF. ICLR 2025  
> * Know Your Limits: A Survey of Abstention in Large Language Models. TACL

**Foundations: selective prediction, deferral, process supervision, UQ**

> * Geifman & El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017  
> * Kamath, Jia & Liang. Selective Question Answering under Domain Shift. ACL 2020  
> * Madras, Pitassi & Zemel. Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer. NeurIPS 2018  
> * Mozannar & Sontag. Consistent Estimators for Learning to Defer to an Expert. ICML 2020  
> * Lightman et al. Let's Verify Step by Step. ICLR 2024  
> * Quach et al. Conformal Language Modeling. ICLR 2024  
> * Farquhar et al. Detecting hallucinations in large language models using semantic entropy. Nature, 2024  
> * Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know? arXiv:2506.18183  
> * LM-Polygraph: benchmarking UQ methods for LLMs. TACL; github.com/IINemo/lm-polygraph

**Infrastructure**

> * Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide (2026). arXiv:2601.09527  
> * vLLM. Extracting hidden states from vLLM. vllm.ai/blog/2026-03-30-extract-hidden-states

## **13\. Summary of changes from revision 1**

For the supervisor's convenience, the substantive changes and why they were made:

| Change | Reason |
| :---- | :---- |
| Scope tiered into committed (RQ1–2) / gated (RQ3) / appendix (RQ4–5); estimator families reduced from seven to five committed | Revision 1 was scoped at master's-to-doctoral size: 5 RQs, 7 families, 5 models, 3 environments, 2 new artifacts in 38 weeks |
| Action-split success rates added to §5; model selection gated on action SR (§6.7) | Two published open-weight models score **0.00%** on action tasks. Selecting on overall SR risks degenerate labels at exactly the turns under study |
| Statistical protocol specified (§6.6); "CIs over seeds" withdrawn | Seeds resample the model, not the task population; a fixed 150-task set gives ±0.08–0.12 AUROC intervals. This was the largest threat to the thesis being falsifiable |
| Task re-instantiation added (§6.3) with a Week-1 gate | Task prompts are `{MRN}`-parameterised and graders are per-category, so N may be expandable to 500–1,000+ at zero annotation cost |
| RQ2 comparison restructured: matched budget, mandatory position-only ablation (§6.5) | Writes are terminal, so commitment gating wins by construction unless controlled for |
| Conformal risk control added (§6.5) | Converts a cost curve into a distribution-free guarantee; cheap; positions the work against the nearest competing paper |
| Three related-work rows added (§3): role-stratified CRC, FHIR-AgentBench, Knowing When to Abstain. Classical foundations added to §12 | Revision 1 had a 2026-only bibliography and omitted the closest neighbour to its own aggregation idea |
| Argus ρ=0.705 attribution corrected (§3) | ρ=0.705 is dataset shift with the model fixed; the cross-model result is ρ≈0.08 |
| §6.2 now distinguishes what `refsol.py` already provides from what the project adds | Binary POST checking is pre-existing; claiming it would not survive defence |
| Attention family moved off the committed path; vLLM constraints documented (§8.4) | vLLM exposes no attention weights; hidden-state extraction requires chunked prefill disabled |
| Interactive GPU-hours doubled; total 3,050 → ≈3,550 (§8.3) | 15 held-GPU-hours per week during instrumentation debugging was not realistic |
| Transfer target switched to FHIR-AgentBench, τ²-bench demoted to trajectory-level | τ²-bench needs a second environment *and* a user simulator; FHIR-AgentBench reuses the entire harness |
