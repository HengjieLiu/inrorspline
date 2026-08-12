# LungCT DUAL-INR 1 mm Lung-Mask Revision Plan

## Goal

Add a controlled DIR-Lab 4DCT DUAL-INR revision branch for the 1 mm pTV-style
setting using the fixed T00 lung mask as the optimization domain. This branch
removes semantic/lobe/segmentation losses and tests whether DUAL-INR is a fair
IDIR-style comparator when the image domain and loss domain are made explicit.

Run a small loss-definition pilot first, then choose one branch for the full
10-case 4DCT sweep and optional hyperparameter sweep.

## Fixed Protocol

- Dataset scope: DIR-Lab 4DCT only.
- Cases: `01` through `10` for full scale-up.
- Pilot cases: `01` and `08`.
- Registration direction: T50 to T00.
- Base image setting: canonical pTVReg iso1mm cropped 4DCT images.
- Fixed image: T00.
- Moving image: T50.
- Optimization mask: fixed T00 1 mm lung mask.
- Metric mask: fixed T00 1 mm lung mask.
- Do not use the moving lung mask for optimization.
- Do not use lobe loss, segmentation loss, or semantic mask loss.

The intended new protocol differs from the existing pTV protocol in
`scripts/experiments/lungct_dual_inr_controlled/data.py`, where protocol `P`
currently uses a border-five interior sampling mask and keeps the fixed T00
lung mask as metric-only. This revision should add a clearly tagged 1 mm
lung-mask optimization variant rather than silently changing the existing `P`
semantics.

## DUAL-INR Settings To Preserve

- Use the released DUAL-INR architecture.
- Keep the current omega values:
  - coarse omega: `10`
  - fine omega: `30`
- Keep semantic/lobe/segmentation loss disabled:
  - `mask_weight = 0`
- Keep total training budget: `2500` steps.
- Keep the current controlled multiresolution stage budget unless the pilot
  explicitly documents a change:
  - stage lengths: `400,400,800,900`
  - image factors: `8,4,2,1` for multiresolution branch
- Do not tune omega for this revision branch.

## Loss Pilot Variants

Run both variants on cases `01` and `08` before the full sweep.

### DUAL-native regularization

- Similarity: sampled global NCC on the fixed T00 lung-mask domain.
- Regularization: released DUAL Laplacian and negative-Jacobian terms.
- Semantic/lobe/segmentation loss: disabled.
- Purpose: test DUAL-INR with its native spatial regularization under a
  lung-mask optimization domain.

### IDIR-type BE regularization

- Similarity: sampled global NCC on the fixed T00 lung-mask domain.
- Regularization: IDIR-style coordinate-autograd bending energy.
- Preserve the audited IDIR bending-energy scale convention, including the
  historical batch-size scaling if reusing the IDIR implementation.
- Semantic/lobe/segmentation loss: disabled.
- Purpose: test whether DUAL-INR can be made closer to the IDIR loss family
  for fairer interpretation.

## Required Outputs

For each pilot and full-scale run, record:

- Case ID.
- Protocol tags.
- Source image/mask paths in portable or review-safe form.
- Fixed/moving image shapes and spacing.
- Optimization mask voxel count.
- Metric mask voxel count.
- Stage trace and total step count.
- Native/crop snap TRE.
- Subvoxel TRE.
- Common posthoc regularity metrics.
- Runtime.
- Peak memory.
- Loss branch identifier:
  - `dual_native_reg`
  - `idir_type_be`

The summary must make clear that TRE or regularity magnitude is not used for
technical acceptance of a completed record. Acceptance should be based on
expected artifacts, finite metrics, correct protocol identity, expected stages,
and completed posthoc regularity.

## Pilot Acceptance And Decision Rule

The pilot passes technically only if both loss variants complete cases `01` and
`08`, produce finite metrics, use the fixed T00 1 mm lung mask as both sampling
and metric mask, and write enough provenance to distinguish the branch from the
existing native/full-volume and pTV-border-mask DUAL runs.

After the pilot:

- Choose one loss branch for the full 10-case 4DCT sweep.
- Prefer the IDIR-type BE branch if it is stable and makes the comparison more
  interpretable under the loss-function audit.
- Prefer the DUAL-native branch only if IDIR-type BE is unstable, clearly
  mis-scaled, or too invasive to implement safely.
- Run an optional hyperparameter sweep only after the validated branch is
  selected.

## Interpretation Rule

Compare only within matched protocol and loss domains. Existing LungCT rows
from native IDIR, pTVReg-style spline methods, original DUAL-INR, and this new
1 mm lung-mask DUAL-INR branch are not clean method-only comparisons unless
preprocessing, crop, mask, metric, interpolation, and loss conventions are
explicitly matched or tagged.

Raw alpha or regularization weights are not directly comparable across
DUAL-native and IDIR-type losses because of the audited loss discrepancy in
`docs/offgrid_submission/camera_ready_loss_function_audit_and_standard.md`.

## Notes For The Execution Agent

- Add a new protocol/variant handle rather than mutating existing `P` behavior.
- Keep the existing `N0`, `N1`, `P0`, and `P1` rows reproducible.
- Use fixed T00 lung-mask optimization for the new 1 mm branch.
- Do not introduce moving-mask optimization unless a separate nonstandard
  protocol is explicitly planned.
- Do not launch the full 10-case sweep until the two-case loss pilot is
  inspected.
