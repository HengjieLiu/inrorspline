---
doc_type: experiment_execution_spec
status: approved_for_execution
experiment_id: 2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1
---

# GPU-1 LungCT cuDNN benchmark-off deterministic rerun

## Question

Determine whether `cudnn.benchmark=True, deterministic=False` caused the slow
shape-first observations in P34 by repeating the same 1-cold plus 10-warm
layout with `cudnn.benchmark=False, deterministic=True`. Compare the new warm
cohort with both the original P34 warm cohort and P35's immediate same-shape
repeats.

## Frozen method and sequence

Preserve the P34 canonical pTVReg-style isotropic-1-mm data, pTV local-CC plus
isoTV objective, four adaptive/fixed-50 schedules, Adam seed 1, exact knot
insertion, final-stage-only full-grid materialization, synchronized
`case_load_complete -> dvf_file_ready` endpoint, and physical-GPU-1 controls.

Each dataset/variant arm is one fresh persistent process:

1. case 01 process-cold;
2. case 01 immediate warm repeat;
3. cases 02-10 process-warm, each first occurrence of its matrix.

This produces 44 registrations: four cold QA fields and 40 warm cohort fields.
All rows must record requested and effective cuDNN flags, PyTorch/CUDA/cuDNN
versions, and device identity.

## Comparisons

- P34 reference:
  `lungct_mrdbscp_adaptive_fixed50_gpu1_20260929T170843Z`, using
  `benchmark=True, deterministic=False` and the same 44-row layout.
- P35 reference:
  `lungct_mrdbscp_shape_pair_cache_diagnostic_gpu1_20260929T211725Z`, using
  `benchmark=True, deterministic=False` with adjacent first/repeat pairs.
- Fixed-50 is the primary backend comparison because all observations use
  identical 200-step 4DCT or 250-step COPD schedules. Adaptive comparisons
  must report optimizer-step mismatches.
- Report casewise and aggregate new F/T warm time, prior P34 T/F warm time,
  P35 T/F immediate-repeat time, case-01 cold time, quality deltas, peak
  allocation, and stage steps.

## GPU, restart, and artifact contract

Use physical GPU 1 only. Before every arm require three consecutive empty
NVML samples; monitor at approximately 1 Hz and case boundaries. Never touch a
foreign process. Retain and restart the complete arm after contamination.

Generated material is external under:

- `${ARTIFACT_ROOT}/runs/lungct_mrdbscp_cudnn_ft_gpu1_<timestamp>`
- `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_lungct_mrdbscp_cudnn_ft_gpu1/<run-id>`

## Acceptance

Require the 20-case loader audit, four one-step F/T smokes, 44 clean timing
rows, 44 finite case-specific float32 DVFs, 40 unique warm cohort fields,
300-landmark accounting, exact schedules/materialization traces, requested and
effective F/T flags on every row, complete provenance, accepted P34/P35
references, and no monitor gap over five seconds.
