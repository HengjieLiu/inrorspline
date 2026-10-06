---
doc_type: experiment_technical_log
status: complete
experiment_id: 2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1
---

# Technical log

## Known limitations

- `cudnn.deterministic=True` constrains cuDNN algorithm selection but does not
  assert global bitwise determinism for every CUDA/PyTorch operation.
- Backend flags can change the numerical optimization path. Timing acceptance
  therefore does not substitute for DVF, TRE, regularity, and stopping checks.
- P34, P35, and this rerun are separate processes executed at different times.
  They use the same physical GPU, container, data, schedules, and endpoint,
  but fixed-50 remains the strongest comparison because its work count is
  identical.
- Adaptive stopping can change between backend configurations; every mismatch
  must remain visible in the comparison.
- P34 and P35 artifacts are immutable inputs and will not be regenerated or
  modified.

## Implementation

- The shared LungCT CUDA helper and `run_one.py` now default future unspecified
  runs to `benchmark=False, deterministic=True`. Historical P34/P35 launchers
  pass `True/False` explicitly so their recorded protocol remains reproducible.
- `run_one.py` records requested/effective backend flags and library/device
  provenance in each summary.
- The new wrapper reuses P34's schedules, GPU ownership, idle gate,
  contamination restart, resume, and external-routing contracts while forcing
  `benchmark=False, deterministic=True`.
- The dedicated aggregator validates the backend flags and compares new warm
  rows with accepted P34 warm rows and P35 immediate repeats.

## Execution events

- Run ID: `lungct_mrdbscp_cudnn_ft_gpu1_20260930T032535Z`.
- The 44-row no-execution manifest and focused CPU contracts passed (`40/40`).
- All 20 dataset loaders and four one-step GPU-1 smokes passed. Every smoke
  recorded effective `benchmark=False, deterministic=True`, zero bad monitor
  samples, and maximum monitor gaps of 1.086-1.147 seconds.

## Completed production

- Technical acceptance passed with 44/44 timing rows, 44 finite case-specific
  fields, 40 unique warm cohort fields, four cold QA fields, and zero aggregate
  validation errors.
- Every summary recorded requested and effective
  `benchmark=False, deterministic=True` plus complete library/device metadata.
- All four production arms were accepted on `attempt_00`; none was
  contaminated or retried. Accepted monitor gaps were 1.192-1.199 seconds
  with zero bad samples. Physical GPU 1 was empty after completion.
- External review root:
  `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1/lungct_mrdbscp_cudnn_ft_gpu1_20260930T032535Z/`.

## Timing comparison

| Dataset | Variant | New F/T warm s | P34 T/F warm s | P35 T/F immediate repeat s | P34/new | New/P35 floor | Step mismatches P34/P35 |
|---|---|---:|---:|---:|---:|---:|---:|
| 4DCT | adaptive | 7.589 +/- 1.551 | 57.097 +/- 33.319 | 8.670 +/- 1.502 | 7.52x | 0.88x | 6/6 |
| 4DCT | fixed-50 | 3.958 +/- 0.449 | 48.690 +/- 30.222 | 3.674 +/- 0.717 | 12.30x | 1.08x | 0/0 |
| COPD | adaptive | 12.482 +/- 2.182 | 51.332 +/- 46.893 | 11.157 +/- 2.064 | 4.11x | 1.12x | 7/9 |
| COPD | fixed-50 | 4.689 +/- 0.577 | 36.228 +/- 37.620 | 4.318 +/- 0.487 | 7.73x | 1.09x | 0/0 |

Fixed-50 is decisive: disabling cuDNN benchmarking removed the slow
shape-first cohort while preserving identical work. New F/T warm time was
within 7.7% of the P35 4DCT immediate-repeat floor and within 8.6% for COPD.
Case-01 process-cold fixed-50 time also fell from 57.701 to 4.670 seconds for
4DCT and from 101.004 to 5.615 seconds for COPD. The prior P34 use of the term
"warm" meant process-warm, not shape-warm; its slow cases were dominated by
new-shape cuDNN autotuning.

Adaptive timing moved in the same direction, but its cross-run timing ratios
remain descriptive because stopping steps differed in 6/10 4DCT and 7/10 COPD
cases versus P34.

## Quality check

Warm native-crop snap TRE under F/T was 0.9188/0.9204 mm for 4DCT
adaptive/fixed-50 and 1.0210/1.0349 mm for COPD. Fixed-50 F/T minus P34 T/F
mean TRE was -0.0022 mm for 4DCT and -0.0120 mm for COPD; versus P35 immediate
repeats it was -0.0020 and -0.0017 mm. These small aggregate differences do
not establish bitwise equivalence, but no material cohort-quality regression
was observed.

## Decision

Use `benchmark=False, deterministic=True` for future canonical LungCT timing
unless a separately declared backend study requires another setting. Do not
reinterpret P34's process-warm timing as steady-state per-case runtime; retain
it as evidence of T/F shape-first behavior. P36 is the matched backend-fixed
runtime record, and P35 remains the independent immediate-repeat floor.

Generated outputs remain under `${ARTIFACT_ROOT}`.
