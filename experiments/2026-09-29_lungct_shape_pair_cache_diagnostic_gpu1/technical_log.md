---
doc_type: experiment_technical_log
status: complete
experiment_id: 2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1
---

# Technical log

## Known limitations

- Adjacent repetition tests the effect of already-established state for an
  identical matrix, but does not by itself identify whether the mechanism is
  cuDNN plan selection, CUDA module loading, allocator state, kernel geometry,
  or another backend effect.
- Case 01 first execution is process-cold; cases 02-10 first executions are
  process-warm and shape-first. Those categories remain explicit.
- Adaptive stopping may produce different step counts between repetitions.
  Fixed-50 is therefore the primary causal diagnostic for backend-state
  effects; adaptive results remain descriptive unless paired steps match.
- Repeated fields are technical timing/QA records, not new independent quality
  samples.

## Implementation

- The diagnostic wrapper reuses the accepted P34 controller's GPU ownership,
  idle-gate, process-identity, restart, schedule, and artifact-routing logic.
- P34 was minimally parameterized with worker-script, aggregator-script, and
  rows-per-arm hooks; its defaults and accepted manifest remain unchanged.
- The diagnostic has a separate immutable run ID, provenance snapshot,
  aggregator, review report, and tests.

## Execution events

- 2026-09-29 21:17 UTC: source compilation and focused controller/diagnostic/
  artifact-path tests passed (`34/34`). The immutable dry run contained 80
  unique registrations in 40 adjacent pairs.
- 2026-09-29 21:24 UTC: all 20 loaders and four fresh one-step GPU-1 smokes
  passed. Accepted smoke monitor intervals had zero bad samples and maximum
  gaps of 1.18-1.19 seconds.
- 2026-09-29 21:25 UTC: the initial production worker exited during argument
  parsing because `run_one.py` had not yet admitted the diagnostic-only
  `shape_first/shape_repeat` timing labels. No registration or CUDA work began.
  The controller was interrupted after six retained parse-only attempts. The
  timing-label enum was extended without changing the optimizer or P34 labels;
  production will resume as a fresh complete-arm attempt.

## Completed production

- Run ID:
  `lungct_mrdbscp_shape_pair_cache_diagnostic_gpu1_20260929T211725Z`.
- External review root:
  `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1/lungct_mrdbscp_shape_pair_cache_diagnostic_gpu1_20260929T211725Z/`.
- Technical acceptance passed with 80/80 finite fields, 40/40 complete adjacent
  pairs, zero validation errors, and 300-landmark accounting for every field.
- All four accepted production arms had zero bad monitor samples. Maximum
  monitor gaps were 1.199-1.202 seconds. Physical GPU 1 was empty after the
  controller exited.
- The accepted 4DCT adaptive production work is `attempt_07`; attempts 00-05
  were retained argument-parse failures before any registration/CUDA work,
  and the controller was interrupted while opening attempt 06. The other
  three accepted arms are `attempt_00`. No accepted arm was contaminated or
  retried after registration began.

## Timing result

Synchronized loaded-input to serialized-DVF timing, mean +/- sample SD:

| Dataset | Variant | Shape-first s | Immediate repeat s | Ratio of means | Step-changed pairs |
|---|---|---:|---:|---:|---:|
| 4DCT | adaptive | 68.525 +/- 32.267 | 8.670 +/- 1.502 | 7.90x | 3/10 |
| 4DCT | fixed-50 | 76.902 +/- 39.292 | 3.674 +/- 0.717 | 20.93x | 0/10 |
| COPD | adaptive | 70.924 +/- 61.399 | 11.157 +/- 2.064 | 6.36x | 7/10 |
| COPD | fixed-50 | 62.669 +/- 60.096 | 4.318 +/- 0.487 | 14.51x | 0/10 |

Fixed-50 supplies the clean causal diagnostic: both observations perform the
same 200 4DCT or 250 COPD optimizer steps. The adaptive ratios are descriptive
because stopping changed in 3/10 4DCT and 7/10 COPD pairs.

The case pattern is not explained well by full-volume voxel count. Pearson
correlations between voxel count and shape-first time were 0.258/0.270 for
4DCT adaptive/fixed-50 and 0.338/0.332 for COPD. In contrast, the casewise
shape-first pattern agreed almost perfectly between adaptive and fixed-50:
`r=0.991` for 4DCT and `r=0.999` for COPD. Mean peak allocation fell by
2.876x for 4DCT and 2.873x for COPD on the immediate repeat.

Same-seed repetitions were not numerically identical under the current
`benchmark=True, deterministic=False` backend. Maximum absolute native-crop
subvoxel TRE drift was 0.0030/0.0011 mm for 4DCT adaptive/fixed-50 and
0.2344/0.0497 mm for COPD. The fixed-50 result is especially relevant because
the schedule and step count are identical; backend remediation must therefore
gate on field/quality reproducibility as well as time.

This proves that already-established same-shape process/backend state dominates
the observed first/repeat gap. It does not by itself prove a specific cache
owner. The leading mechanism is cuDNN shape-specific autotuning: the LungCT
runner explicitly selects `cudnn.benchmark=True, deterministic=False`, and the
existing controlled OASIS backend ablation removed most cold overhead with
`benchmark=False` while leaving warm time nearly unchanged. A Lung-specific
backend-flag A/B remains required before attributing the effect solely to
cuDNN.

## Next decision

Use fixed-50 to test three backend configurations on representative slow and
already-fast shape-first cases: benchmark false/deterministic true, benchmark
false/deterministic false, and the current benchmark true/deterministic false.
Record first/repeat time, per-stage timing, peak allocation, DVF equivalence,
TRE, and regularity. Prefer benchmark false/deterministic true if it removes
the shape-first penalty without material warm or quality regression. Explicit
same-shape priming is deployment-only because it moves rather than removes the
cost; padded shape bucketing is a later option and requires objective/border
revalidation.

Generated fields, logs, monitor streams, manifests, and reports remain under
`${ARTIFACT_ROOT}`.
