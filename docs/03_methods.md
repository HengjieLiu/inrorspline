---
doc_type: methods_plan
status: active_draft
---

# Methods

## Method Families

- `IDIR`: coordinate-network deformation model. Use original-style loss/regularization where possible, and keep alpha scales separate from spline methods.
- `SINR`: hybrid INR/spline method. Compare `cps4` and `cps2` when feasible.
- `DirectCP`: direct optimization of spline/control-point parameters without the INR component.
- Reference methods: OASIS learned baselines and Greedy; LungCT pTVReg official/reproduced references.

## OASIS Brain Protocol

Main target:

- seed-3 shuffled first10 pairs.
- Canonical foreground/domain mask: raw image-positive foreground, `image > 0`.
  `seg > 0` is a label/comparison mask unless a run explicitly documents it as
  an override.
- Main methods: IDIR original, SINR cps4, SINR cps2, DirectCP cps4.
- Primary comparison: Dice or HD95 versus bending energy, diffusion energy, and log-Jacobian spread.
- Baseline dots: saved DL methods and Greedy variants where same-pipeline deformation metrics can be computed.

Known required consistency items:

- Pair order and case IDs.
- Fixed/moving direction.
- Label set.
- OASIS foreground rule: default to raw `image > 0`, not `seg > 0`.
- HD95/ASD missing-label exclusion rule.
- Physical unit convention for displacement fields.
- Dense evaluation resolution and foreground/mask definition.
- `align_corners` and interpolation conventions.

## LungCT / 4DCT Protocol

Main target:

- 10 4DCT pairs.
- Benchmark references: pTVReg official/reproduced, clearly labeled by preprocessing protocol.
- Method variants: IDIR, SINR, single-stage DirectCP cps4 negative control, and multistage DirectCP such as `32-16-8-4`.

Required warning:

Do not compare LungCT rows as method-only results unless the input preprocessing protocol is matched. Native full-resolution IDIR, pTVReg landmark-cropped 1 mm runs, and pTV-preprocessed SINR/DirectCP see different optimization problems.

## Regularity Metrics

Use a consistent posthoc metric pipeline where possible:

- diffusion energy
- bending energy
- log-Jacobian standard deviation
- folding percentage / `J <= 0`
- NDV as supplementary, not the sole regularity endpoint

For LungCT, report TRE in physical mm and keep snap/sub-voxel conventions explicit.
