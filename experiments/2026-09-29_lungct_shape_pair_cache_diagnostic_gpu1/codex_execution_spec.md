---
doc_type: experiment_execution_spec
status: approved_for_execution
experiment_id: 2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1
---

# LungCT adjacent shape-pair cache diagnostic

## Question

Determine how much of the casewise LungCT runtime variation disappears when
every unique matrix is executed twice consecutively in the same process. This
is a separate diagnostic; the accepted P34 fields and reports remain
immutable.

## Frozen method

Use the same four P34 arms, canonical pTVReg-style isotropic-1-mm data,
objective, schedules, seed, exact knot insertion, final-only full-grid
materialization, timing endpoint, physical GPU-1 idle gate, NVML monitoring,
contamination restart, container, and external artifact routing.

The schedules remain:

- 4DCT adaptive: controls `32/16/8/4`, downsampling `8/4/2/1`, caps
  `1400/400/400/400`, LR `.01/.003/.003/.003`, minima `210/60/60/60`, and
  patience `50/25/25/25`.
- 4DCT fixed-50: the same geometry/LR and `50/50/50/50` steps.
- COPD adaptive: controls `64/32/16/8/4`, downsampling `16/8/4/2/1`, caps
  `800/600/400/400/400`, LR `.03/.03/.01/.01/.003`, minima
  `120/90/60/60/60`, and patience `50/38/25/25/25`.
- COPD fixed-50: the same geometry/LR and `50/50/50/50/50` steps.

## Diagnostic sequence

Each dataset/variant arm is one fresh persistent process. For case 01 through
case 10, run an adjacent pair:

1. `shape_first`: first occurrence of that case matrix in the arm;
2. reset controls, optimizer, and seed;
3. `shape_repeat`: immediate repeat of the identical matrix.

This produces 80 registrations and 40 pairs. Case 01 `shape_first` is both
process-cold and shape-first. Cases 02-10 `shape_first` are process-warm but
shape-first. Fixed-50 is the clean backend-state diagnosis because paired step
counts are identical. Adaptive rows retain natural stopping and must report
any paired step-count change as a timing confounder.

## Outputs and acceptance

Generated material is external under:

- `${ARTIFACT_ROOT}/runs/lungct_mrdbscp_shape_pair_cache_diagnostic_gpu1_<timestamp>`
- `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_lungct_shape_pair_cache_diagnostic_gpu1/<run-id>`

Acceptance requires 80 clean finite fields, 40 complete adjacent pairs,
300-landmark accounting, correct schedules/materialization traces, identical
fixed-50 first/repeat step counts, complete provenance, and four accepted arm
monitor intervals with no foreign process/query error and gaps no longer than
five seconds. Report casewise matrix sizes, first/repeat timing, paired
reduction/speedup, optimizer steps, and aggregate results. Do not call the
diagnosis a cache mechanism proof; use it to choose the next targeted fix.
