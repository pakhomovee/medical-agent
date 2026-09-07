# Gate G2 — Qwen3-8B, Colab A100, 2026-09-07

**PASS** on the no-thinking tier. 50 tasks (5 per template), graded against `refsol.py`.

| | no-thinking | thinking + `--strip-think` |
|---|---|---|
| **action SR** | **52.0%** (threshold ≥40%) | 0.0% |
| query SR | 12.0% | 24.0% |
| overall SR | 32.0% | 12.0% |
| schema-valid POSTs | 100% (10/10) | 100% (10/10) |
| invalid actions | 2% | 22% |
| round/context limit | 28% | 8% |
| mean turns | 2.72 | 1.88 |
| tokens generated | 13,711 | 82,226 (6.0x) |

## The scaffold effect dominates everything

Action success moves **0% -> 52% from one flag**, holding the model, tasks and
environment fixed. For comparison, the PSB 2026 paper's scaffold rewrite moved overall
success 69.67% -> 91%, a 21-point swing. This is 52 points on the action split.

This settles plan §8 open decision 2: scaffold tier is not an optional variable to sweep
if time allows, it is the largest single factor in the error rate and must be a declared,
frozen part of any reported configuration.

## Failure stratification (plan §4.4)

| | no-thinking | thinking |
|---|---|---|
| correct | 16 | 6 |
| **prose instead of a value** | 5 | **21** |
| wrong value | 8 | 7 |
| no answer — invalid/truncated | 1 | 11 |
| no answer — round/context limit | 14 | 4 |

**42% of the thinking tier's episodes fail on output format, not clinical judgement.**
Example, task7_2: the model retrieves the correct value and answers

    FINISH(["The most recent CBG for patient S2197736 is 87.0 mg/dL, recorded on ..."])

where the grader wants `[87.0]`. Clinically right, graded wrong.

The reverse also appears. task2 (age from birthdate) scores **100% with reasoning and 20%
without** -- reasoning fixes the arithmetic, then the answer is thrown away on format.

Reporting a single success rate over this mixture would be close to meaningless, which is
the argument §4.4 makes. It now has data behind it.

## Per-template action SR (no-thinking)

| template | SR | note |
|---|---|---|
| task3 record BP | 100% | single unconditional POST |
| task8 order referral | 100% | single unconditional POST |
| task5 magnesium, conditional | 60% | |
| task9 potassium + paired lab | 0% | conditional, 2 POSTs, 0% completion |
| task10 HbA1C conditional order | 0% | completes but answers `[-1]` |

The label distribution is non-degenerate, which is what G2 had to establish: unconditional
writes near-ceiling, conditional writes at floor. The uncertainty signal, if there is one,
lives in that gap.

## Known error modes worth carrying into the kappa audit

- **Type mismatch.** task2_1 expected `[60]`, model returned `["60"]` -- graded wrong. The
  v2 authors documented this class ("returning a number as a string").
- **Malformed patient reference.** task7 queried `patient=Patient/2823623`, dropping the
  `S` from MRN `S2823623` and adding a `Patient/` prefix. Zero results, `FINISH([])`.
- **Round and context limits bind.** 28% of the no-thinking tier ran out of turns
  (`max_round=5`) or context (8192). task6 hit 5.00 turns at 0% completion.

## Configuration

Qwen3-8B, bfloat16, Colab A100 (sm80). vLLM `--max-model-len 8192 --enable-prefix-caching
--no-enable-chunked-prefill`. FHIR on :8089 (Colab's node process owns :8080).
`max_round=5`, temperature 0. Logprob determinism measured at exactly 0.000e+00 over 8
identical requests -- same as an RTX 5090, so no nondeterminism noise floor on either.
