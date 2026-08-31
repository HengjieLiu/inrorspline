---
experiment_id: 2026-08-30_learn2breath_mrdbscp_p2e_pairlocal
status: all20_two_window_dedicated4gpu_complete
date: 2026-08-30
dataset: learn2breath_challenge_like_copd10_pairlocal_1p5mm
reference_cohort: copd_mr_d_bscp_p2e_iso1mm
---

# MR-D-BSCP P2e: Learn2Breath-style 1.5 mm port

## Scope

This experiment ports the selected historical COPD P2e optimisation mechanics
to the shared-grid Learn2Breath-style COPD data. It is a **full-grid, 1.5-mm
CT-only adaptation**, not a reproduction of the historical 1-mm landmark-crop
result.

The initial technical audit was COPD8, represented by `case_111`. Following
review, the execution scope is now a controlled 40-registration comparison:
ten COPD and ten Learn2Breath validation pairs under two reported-only CT
windows. No window is selected from this cohort result.

The initial single-case audit exposed a preprocessing implementation error:
the historical raw-value window was applied directly to physical HU. That
superseded run and the correction rationale are recorded in
[`copd8_audit_report.md`](copd8_audit_report.md). Its corrected result is
retained as provenance; COPD8 is rerun in the finalized cohort for uniform
timing and output metadata.

## Locked registration protocol

| Item | P2e adaptation |
| --- | --- |
| Fixed / moving | phase 1 INSP / phase 2 EXP |
| Registration grid | Shared `224 x 192 x 224` fixed grid, 1.5 mm isotropic; no crop or resampling |
| Intensity profiles | `p2e_equiv_hu`: physical HU `[-920,-100]`, equivalent to historical stored `[80,900]`; `l2b_lung_hu`: Learn2Breath lung window `[-1000,200]` |
| Optimisation domain | Full image with only a five-voxel border mask |
| Similarity | pTV local CC, sigma 2.1 mm, radius `2*sigma`, sigma floor 0.8 pixels |
| Regularisation | Physical isoTV, weight 0.11, `csqrt=0.005` |
| CP spacing in full-grid voxels | `64 -> 32 -> 16 -> 8 -> 4` |
| Image downsampling | `16 -> 8 -> 4 -> 2 -> 1` |
| Stage epochs | `400, 400, 400, 400, 900` (2,500 total) |
| Optimiser / seed | Adam `1e-3` / seed 1 |
| Sampling / prolongation | historical `align_corners=True`; exact crop-aligned dyadic cubic B-spline knot insertion |
| Exported field | `float32 [3,X,Y,Z]`, XYZ voxel components, `moving(fixed + u)` |

The earlier proposed 32x image-pyramid extension is explicitly **inactive**:
it would be a new six-stage method, whereas this run is the five-stage P2e
port above.

## Historical target and non-equivalence

The source method is the completed 1-mm pTV-style crop run documented in
[`../2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity/codex_execution_spec.md`](../2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity/codex_execution_spec.md).
The schedule, loss family, interpolation convention, and historical CT
window are intentionally retained. The input grid, physical spacing, field
domain, landmark convention, and evaluation context differ.

Both products are read as physical-HU NIfTI inputs. COPD provenance declares
`HU = stored - 1000`, making `[-920,-100]` the exact physical-HU expression of
historical P2e stored `[80,900]`. Validation does not expose raw storage, so
both profiles are applied directly and explicitly in HU. Direct `[80,900]` HU
is excluded because it collapses most voxels to zero.

Landmarks, R231 masks, and lobe labels are excluded from the registration
loss. COPD landmarks are used only for post-registration physical trilinear
TRE. R231 masks are QC-only. Learn2Breath lobe labels are reserved for the
unchanged validation Dice/folding evaluator after the COPD audit is approved.

## Required audit gates

Before optimisation, the runner verifies the pair shape, spacing, affine,
phase direction, HU window occupancy, all 300 paired-landmark IDs and bounds,
baseline physical TRE, R231 geometry, zero and known-shift field behaviour,
and NIfTI field export/reload. It writes the preflight report with the run
artifacts, rather than embedding local data paths in this repository.

COPD8 outputs include a per-landmark CSV, physical trilinear TRE, stage
runtime, Jacobian/folding diagnostics, and the exported fixed-grid DVF. The
authorized two-window rollout also evaluates the ten validation cases with the
unchanged lobe Dice/folding evaluator.

## Registration visual audit

[`registration_audit.ipynb`](registration_audit.ipynb) is the primary review
surface. After the COPD8 DVF exists, it validates the R231/CT grids and
renders one inline 2x9 coronal figure:

- CT row: moving EXP, warped moving EXP, fixed INSP at fixed-R231 25%, 50%,
  and 75% anterior--posterior positions.
- R231 row: moving mask, nearest-neighbour warped moving mask, fixed mask at
  the same positions.

The warped CT and mask use the P2e `align_corners=True` sampling convention.
The view exactly follows the paired mask audit: `origin="lower"`, superior at
the top, and patient right on screen left. The notebook deliberately does not
save a standalone figure by default.

## Dedicated four-GPU rollout

GPU 0 runs all COPD `p2e_equiv_hu` cases, GPU 1 runs all COPD
`l2b_lung_hu` cases, GPU 2 runs all validation `p2e_equiv_hu` cases, and GPU
3 runs all validation `l2b_lung_hu` cases. Each persistent worker's first case
is cold; its remaining nine cases are warm. Every run records synchronized
optimizer-only time, full load-to-evaluation time, worker case time, worker
total, and the four-worker makespan. The four cold outputs must pass before
their workers continue to warm jobs.

## Completed two-window audit

All 40 preflights and registrations completed on 2026-08-30. The cold gates
passed finite field and metric checks before the controller released the warm
queues. No window is selected from these reported-only results.

| Cohort | Profile | Main result | Challenge whole-grid folding | Cold full-case s | Warm full-case s (mean +/- SD) |
| --- | --- | ---: | ---: | ---: | ---: |
| COPD | `p2e_equiv_hu` | mean TRE 1.4683 mm | 0.2444% | 394.7 | 311.7 +/- 121.4 |
| COPD | `l2b_lung_hu` | mean TRE 1.7598 mm | 0.7905% | 411.7 | 268.5 +/- 96.0 |
| Validation | `p2e_equiv_hu` | mean lobe Dice 0.9724 | 0.2767% | 446.7 | 327.1 +/- 132.8 |
| Validation | `l2b_lung_hu` | mean lobe Dice 0.9700 | 0.5832% | 446.6 | 326.3 +/- 134.1 |

Validation aggregate Dice by lobe label `(8, 16, 32, 64, 128)` was
`(0.9833, 0.9750, 0.9791, 0.9494, 0.9751)` for `p2e_equiv_hu` and
`(0.9815, 0.9711, 0.9773, 0.9467, 0.9735)` for `l2b_lung_hu`. The
four-worker makespan was 3,395.8 s (56.6 min). The external aggregate retains
all per-case TRE/Dice/folding values, all five per-lobe Dice values, profile
provenance, and cold/warm optimizer/full/worker timings.

[`copd_all10_audit.ipynb`](copd_all10_audit.ipynb) and
[`validation_all10_audit.ipynb`](validation_all10_audit.ipynb) were executed
in Docker with 20 inline 2x9 panels each and no cell errors. COPD shows the
R231 masks; validation shows discrete lobe masks warped with the challenge
sampler. Both retain the accepted coronal orientation and do not create
standalone figure files.
