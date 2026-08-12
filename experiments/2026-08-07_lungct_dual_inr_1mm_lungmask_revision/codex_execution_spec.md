---
doc_type: codex_execution_spec
status: full_complete
experiment_id: lungct_dual_inr_1mm_lungmask_revision_v1
date_planned: 2026-08-07
---

# DUAL-INR 1 mm Lung-Mask Revision Execution Specification

## Scope

Implement, pilot, and scale up a DUAL-INR DIR-Lab 4DCT branch on the canonical
pTVReg iso1mm image domain using the fixed T00 lung mask as both the
optimization sampling mask and posthoc metric mask.

Pilot cases `01` and `08` passed technical inspection on 2026-08-07.  The full
ten-case sweep completed for `PL1D` and `PL1B` on 2026-08-07.

## Variants

- `PL1D`: protocol `PL`, image factors `[8,4,2,1]`, loss branch
  `dual_native_reg`.
- `PL1B`: protocol `PL`, image factors `[8,4,2,1]`, loss branch
  `idir_type_be`.

Both variants use the released DUAL architecture, coarse/fine omega `10/30`,
semantic mask weight `0`, stage lengths `[400,400,800,900]`, and total
training budget `2500` steps.

## Protocol Contract

- Dataset: DIR-Lab 4DCT cases.
- Direction: T50 moving image to T00 fixed image.
- Image domain: canonical `pTVReg_iso1mm/4DCT/caseXX`.
- Sampling mask: fixed T00 1 mm lung mask only.
- Metric mask: fixed T00 1 mm lung mask only.
- Moving T50 mask: not used for optimization.
- Semantic, lobe, and segmentation losses: disabled.

This is distinct from protocol `P`, which keeps the fixed T00 lung mask
metric-only and samples from the five-voxel pTV crop interior.

## Launch Contract

Dry-run:

```bash
python -m scripts.experiments.lungct_dual_inr_controlled.launch_lungmask_revision \
  --stage dry-run
```

Technical smoke:

```bash
python -m scripts.experiments.lungct_dual_inr_controlled.launch_lungmask_revision \
  --stage smoke
```

Pilot:

```bash
python -m scripts.experiments.lungct_dual_inr_controlled.launch_lungmask_revision \
  --stage pilot \
  --cases 1,8
```

Full cohort:

```bash
python -m scripts.experiments.lungct_dual_inr_controlled.launch_lungmask_revision \
  --stage full
```

Use the project Docker container for preprocessing/data-inspection and real
GPU execution. Keep run outputs outside Git and outside data roots.

## Acceptance

Technical acceptance requires expected artifacts, finite history/metrics,
protocol `PL`, fixed T00 lung-mask sampling and metric rules, exact stage
lengths/factors, completed dense-field TRE diagnostics, and completed common
posthoc regularity.

TRE or regularity magnitude is not a completion gate.
