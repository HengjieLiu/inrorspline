---
doc_type: experiment_execution_spec
status: approved_for_execution
experiment_id: 2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1
---

# GPU-1 4DCT/COPD adaptive and fixed-50 MR-D-BSCP study

## Scientific question

Measure the cold/warm runtime and all-case quality of the best P33 adaptive
schedule for each LungCT cohort against a fixed-50-per-stage schedule analogue.
This transfers the OASIS acceleration idea while retaining the LungCT DirectCP
cubic B-spline backend and pTV local-CC plus isoTV objective.

P33 is historical quality/reproducibility context only. There is no fresh
2,500-step anchor, so this study must not claim a matched speedup against the
old anchor timings.

## Frozen protocol

- Data: canonical pTVReg-style isotropic-1-mm 4DCT cases 01-10 and COPD cases
  01-10, resolved from `${LUNGCT_PTVREG_ISO1MM}`.
- Direction and metrics: the established `run_one.py` pTV protocol. Headline
  quality is native-crop snap-to-voxel TRE; native subvoxel and registration-
  grid snap/subvoxel TRE are diagnostics.
- Shared objective: Adam, seed 1, pTV LCC sigma 2.1 mm, isoTV weight 0.055,
  `csqrt=5e-3`, five-voxel border, exact cubic knot insertion, and final-stage-
  only full-grid materialization.
- Adaptive stopping: 25-step check/window intervals and total/data relative
  thresholds `5e-5/2e-5`.
- Physical device: GPU 1 only, exposed to each worker as logical CUDA device 0.

| Dataset | Variant | Controls | Downsampling | Epoch caps/steps | LR | Minimums | Patience |
|---|---|---|---|---|---|---|---|
| 4DCT | adaptive | `32/16/8/4` | `8/4/2/1` | `1400/400/400/400` | `.01/.003/.003/.003` | `210/60/60/60` | `50/25/25/25` |
| 4DCT | fixed50 | `32/16/8/4` | `8/4/2/1` | `50/50/50/50` | `.01/.003/.003/.003` | n/a | disabled |
| COPD | adaptive | `64/32/16/8/4` | `16/8/4/2/1` | `800/600/400/400/400` | `.03/.03/.01/.01/.003` | `120/90/60/60/60` | `50/38/25/25/25` |
| COPD | fixed50 | `64/32/16/8/4` | `16/8/4/2/1` | `50/50/50/50/50` | `.03/.03/.01/.01/.003` | n/a | disabled |

## Execution layout

The launcher is
`scripts/experiments/lungct_no_sweep_repro/launch_mrdbscp_adaptive_fixed50_gpu1.py`.
It provides `dry-run`, `preflight`, `all`, and `report` phases and rejects any
GPU other than physical GPU 1.

Each dataset/variant arm uses one fresh persistent worker and runs:

1. case 01 cold;
2. case 01 again warm;
3. cases 02-10 warm.

Controls, optimizer state, and random seed reset for every registration. Only
process, CUDA, and library state persists. Primary scale is 44 registrations:
four arms times eleven measurements. The 40 warm fields form the quality
cohort; the four cold case-01 fields are QA records.

Generated run material is external:

- `${ARTIFACT_ROOT}/runs/lungct_mrdbscp_adaptive_fixed50_gpu1_<timestamp>`
- `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_lungct_mrdbscp_adaptive_fixed50_gpu1/<run-id>`

The configured `py310sitreg_v1_artifact_store` container is started or
validated in place and is never recreated by this launcher.

## GPU integrity and retry contract

- Before every arm, require three consecutive empty physical-GPU-1 NVML
  compute-process samples.
- Sample at about 1 Hz and explicitly at case and arm boundaries.
- A clean attempt has no foreign process, no query error, and no monitoring
  gap above five seconds.
- On contamination, preserve the attempt, signal only the PID whose namespace
  identity and process start time match the study token, wait for GPU 1 to
  clear, and restart the complete arm in a fresh process.
- Never signal or stop a foreign process.
- Resume only a complete attempt with an accepted verdict and the exact
  expected row count.

## Timing and reporting

Primary timing is synchronized `case_load_complete -> dvf_file_ready`, which
includes optimization, final field materialization, CPU transfer, and dense
DVF serialization but excludes TRE and regularity evaluation. Also report
optimizer-only time, in-memory endpoint, peak allocation, actual stage steps,
the case-01 cold/warm gap, and ten-case warm mean/sample SD.

Quality reporting includes all four TRE conventions, full/foreground
diffusion and bending energy, full/foreground SDlogJ and folding, final loss,
and peak memory. Fixed-50 is compared directly only with its same-dataset
adaptive arm.

## Verification and acceptance

Before production:

1. run Python syntax checks and focused CPU tests;
2. validate loaders/manifests for all 20 cases;
3. execute a no-registration 44-row dry run;
4. pass four one-step GPU-1 smokes behind the same idle/monitor gate.

Technical acceptance requires 44 clean timing rows, 44 finite case-specific
`float32` DVFs, final-only materialization traces, 40 unique warm cohort
results, 300 landmarks in every TRE convention for every row, and complete
source/container/dataset provenance. All ten cases finish even if fixed-50
quality is poor; acceptance is execution validity, not noninferiority.
