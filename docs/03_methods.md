---
doc_type: methods_plan
status: active_draft
---

# Methods

## Implementation Reference

For code-level architecture, training-loss definitions, cubic B-spline
formulas, DirectCP optimization details, and paper-claim crosswalks, see
`docs/implementation/`. This file remains the protocol/method plan; the
implementation folder records how the current repo-local code realizes those
methods.

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
- Fixed/moving convention follows `docs/standards/lungct_protocol_tags.md`:
  DIR-Lab 4DCT uses `T50`/end-exhale as moving and `T00`/end-inhale as fixed
  (`T50 -> T00`); COPD uses `eBHCT` as moving and `iBHCT` as fixed.
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

IDIR bending-energy training loss preserves the public original IDIR scale. In
that implementation, coordinate-second-derivative terms are averaged and then
divided again by the sampled coordinate batch size. With the default IDIR batch
size of 10,000, the reported training BE is therefore downscaled by 10,000
relative to the pre-divide mean BE. Keep this behavior for `idir_original`
comparability, but do not compare raw IDIR alpha values directly against
SINR/DirectCP alpha values. Prefer `alpha * training_reg_loss` and common
posthoc regularity metrics when comparing regularization strength.

For reporting, use native analytic BE and diffusion by default. For IDIR this
means coordinate-autograd derivatives of the dense coordinate network: Hessian
BE and Jacobian diffusion. For SINR and DirectCP spline variants this means
cubic B-spline analytic derivatives sampled at voxel centers: second
derivatives for BE and first derivatives for diffusion.

For LungCT, report TRE in physical mm and keep snap/sub-voxel conventions explicit.
