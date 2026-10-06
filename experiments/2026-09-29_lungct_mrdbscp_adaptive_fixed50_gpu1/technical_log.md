---
doc_type: experiment_technical_log
status: prelaunch
experiment_id: 2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1
---

# Technical log

## Known limitations and interpretation guardrails

- There is no fresh 2,500-step anchor. P33 timings used a different
  cold/warm layout and mixed physical GPUs, so they are historical context and
  cannot support a matched runtime speedup claim.
- Fixed-50 is an acceleration stress test, not a prespecified quality-
  noninferiority arm. All cases remain reportable even when quality is poor.
- Results are specific to canonical pTVReg-style isotropic-1-mm crops, the pTV
  local-CC plus isoTV objective, the established landmark conventions, and
  this container/runtime. They are not native/full-volume IDIR comparisons.
- A warm worker preserves process/CUDA/library caches only. Model controls,
  optimizer state, and RNG seed reset for every registration.
- Cold-versus-warm inference is descriptive because each arm has one cold
  observation (case 01) and one same-case warm repeat.

## Implementation record

- New host controller:
  `scripts/experiments/lungct_no_sweep_repro/launch_mrdbscp_adaptive_fixed50_gpu1.py`.
- New acceptance aggregator:
  `scripts/experiments/lungct_no_sweep_repro/aggregate_mrdbscp_adaptive_fixed50_gpu1.py`.
- Existing `run_one.py` optimizer/objective implementation is reused without
  changing the completed P33 launcher or aggregator.
- Host controller is standard-library-only; dependency-sensitive loading,
  registration, and aggregation execute in the configured Docker container.
- Every attempt has a token-bound namespace PID identity. Only that verified
  worker may be terminated after contamination or timeout.

## Execution events

- 2026-09-29 17:08 UTC: focused launcher/artifact-path CPU suite passed
  `30/30`; source compilation passed for the launcher, aggregator, and focused
  test module. The configured runtime lacks `pytest`, so the standard-library
  tests ran in the host test environment; all data/GPU work remained in the
  configured container.
- 2026-09-29 17:09 UTC: immutable no-execution dry run passed with 44 primary
  measurement rows and 40 warm cohort rows at the external run/review roots.
- 2026-09-29 17:10 UTC: first smoke attempt failed before CUDA initialization
  because `docker exec --user 1009:1009` discarded the host user's
  supplementary NVIDIA-device group. The failed attempt was retained. The
  worker wrapper was corrected to drop from root to UID/GID `1009:1009`, add
  only supplementary GIDs `1009,1020`, and clear inheritable/ambient
  capabilities. A probe verified CUDA availability and `1009:1009` artifact
  ownership; the temporary probe was removed.
- 2026-09-29 17:19 UTC: loader audit passed all 20 cases and verified 300
  landmarks per case (6,000 total). Four fresh one-step smokes passed on
  physical GPU 1. Accepted attempts had no query failures or foreign
  processes; maximum monitor gaps were 1.18-1.20 seconds.

- 2026-09-29 17:38 UTC: production completed all four arms and 44/44 primary
  registrations with no production retry. All accepted arms had zero bad NVML
  samples; maximum gaps were 1.199-1.200 seconds. GPU 1 returned to idle.
- Strict aggregation passed with zero errors: 44 finite case-specific
  `float32` DVFs, 40 unique warm cohort fields, four cold QA fields, correct
  final-only materialization traces, 300 landmarks per metric/case, exact
  fixed-50 stage counts, valid adaptive min/cap counts, and complete
  source/container/dataset provenance.
- Warm `case_load_complete -> dvf_file_ready` mean +/- sample SD was
  `57.097 +/- 33.319 s` (4DCT adaptive), `48.690 +/- 30.222 s` (4DCT
  fixed-50), `51.332 +/- 46.893 s` (COPD adaptive), and
  `36.228 +/- 37.620 s` (COPD fixed-50). Fixed-50/adaptive speedups were
  `1.17x` and `1.42x`, respectively.
- Warm native-crop snap TRE mean +/- sample SD was `0.9209 +/- 0.1587 mm`
  (4DCT adaptive), `0.9226 +/- 0.1571 mm` (4DCT fixed-50),
  `1.0054 +/- 0.6673 mm` (COPD adaptive), and `1.0469 +/- 0.6970 mm`
  (COPD fixed-50). Fixed-50 minus adaptive was `+0.0017 mm` for 4DCT and
  `+0.0415 mm` for COPD.

Generated logs, manifests, monitoring streams, fields, tables, and reports are
under `${ARTIFACT_ROOT}` and were not copied into this source tree. The
canonical review entry is
`${ARTIFACT_ROOT}/repo_experiments/2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1/lungct_mrdbscp_adaptive_fixed50_gpu1_20260929T170843Z/review_report.md`.
