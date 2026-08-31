---
experiment_id: 2026-08-30_learn2breath_fireants_greedy_two_window_audit
status: implementation_and_preflight
date: 2026-08-30
dataset: challenge_like_copd10_pairlocal_and_learn2breath_validation_1p5mm
---

# FireANTs Greedy CT-only two-window audit

## Scope

This is a new 40-registration FireANTs audit. It reports Greedy local-NCC on
the ten challenge-style COPD pairs and ten Learn2Breath validation pairs,
each under two physical-HU windows. It then combines the new rows with the
completed, immutable P2e audit, giving two methods by two windows per case.

The P2e results are provenance inputs only: they are never rerun or modified.
No method/window is selected and no challenge submission is produced.

## Locked FireANTs protocol

| Item | Value |
| --- | --- |
| Fixed / moving | INSP phase 1 / EXP phase 2 |
| Registration grid | shared `224 x 192 x 224`, 1.5 mm isotropic; no crop or resampling |
| Optimisation input | scalar CT only; no labels, R231 masks, or landmarks |
| Algorithm / loss | FireANTs Greedy / local normalized cross-correlation |
| Image scales / iterations | `16 -> 8 -> 4 -> 2 -> 1` / `200,200,200,200,200` |
| NCC | rectangular kernel 5 |
| Optimiser | Adam, LR 0.2, compositive deformation |
| Smoothing / convergence | warp 0.25, gradient 0.5, tolerance `1e-4`, 10 tolerance iterations |
| Affine | identity initial affine |
| Exported field | `float32 [3,X,Y,Z]`, XYZ voxel components, `moving(fixed + u)` |

The two reported physical-HU profiles are:

- `p2e_equiv_hu`: clip `[-920,-100]`, then `(HU + 920) / 820`; this is the
  physical-HU equivalent of historical P2e's stored-value `[80,900]` rule.
- `l2b_lung_hu`: clip `[-1000,200]`, then `(HU + 1000) / 1200`.

The legacy FireANTs direct `[80,900]` HU normalizer is explicitly excluded.

## Four persistent GPU lanes

| GPU | Cohort | Profile | Cases |
| ---: | --- | --- | --- |
| 0 | COPD | `p2e_equiv_hu` | `case_104` through `case_113` |
| 1 | COPD | `l2b_lung_hu` | `case_104` through `case_113` |
| 2 | Validation | `p2e_equiv_hu` | `NLST_0001` through `NLST_0010` |
| 3 | Validation | `l2b_lung_hu` | `NLST_0001` through `NLST_0010` |

Each persistent worker runs serially. Its first case is cold, the remaining
nine are warm, and all four warm queues wait until every cold field has passed
finite-field and cohort-metric gates.

## Metrics and QC

COPD uses physical trilinear fixed-grid TRE for all 300 landmarks, together
with baseline/mean/median/p95/max TRE and Learn2Breath challenge whole-grid
folding. Validation uses per-lobe Dice for labels `8,16,32,64,128`, mean Dice,
challenge whole-grid folding, and fixed-lobe folding.

`copd_comparison_audit.ipynb` and `validation_comparison_audit.ipynb` are the
primary qualitative review surfaces. They display all cases, methods, and
windows inline as accepted-orientation `2 x 9` coronal panels. COPD displays
R231 masks; validation displays discrete lobe labels. FireANTs COPD panels use
the challenge sampler, P2e COPD panels preserve established historical P2e
sampling, and all validation lobe panels use the challenge sampler matching
Dice. No standalone figures are saved.

## Runtime recovery provenance

The original CUDA 12.1 FireANTs image cannot allocate a full grid on physical
GPU 0. The same FireANTs source passed the exact recovery smoke on the CUDA
12.1 `fireants-devhj:v1_0` base. The four-lane FireANTs launch therefore uses
that base only for GPU 0 and retains the original image on GPUs 1--3, where it
passed its prior smoke gates. Each summary records its exact runtime
environment; this base-image asymmetry is a protocol caveat, not a method or
window comparison.
