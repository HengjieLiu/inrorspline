# Camera-Ready Loss Function Audit and Suggested Standard

Status: audit plus recommendation, not final policy.

Scope: this note documents the current loss/similarity/regularization choices in
the repo-local paper experiments and proposes a provisional camera-ready standard
for review. It does not change code, relaunch experiments, or enforce policy.

## Executive Summary

The submitted paper states the registration objective as

```text
loss = L_sim(M o phi, F) + lambda R(u)
```

and says that IDIR/SINR keep the official image-similarity losses and optimizer
settings. That statement is directionally correct, but it is not yet precise
enough for the camera-ready/reproducible-code release. The current code uses
several objective families that must remain explicitly named:

- OASIS IDIR: sampled global NCC over fixed raw `image > 0` coordinates plus
  original-style IDIR coordinate-autograd bending energy (BE), including the
  extra divide by coordinate batch size.
- OASIS SINR/D-BSCP/MR-D-BSCP: full-image global NCC plus analytic cubic
  B-spline BE. Diffusion is reported as a regularity metric, not used as the
  OASIS spline training regularizer.
- LungCT default IDIR: fixed-phase lung-mask sampled global NCC plus IDIR
  coordinate-autograd BE.
- LungCT pTV-style spline rows: pTV-style local CC (LCC), sigma `2.1 mm`, a
  5-voxel border mask, no lung-mask optimization, plus physical isoTV on control
  values with weight usually `0.11` and `csqrt=5e-3`.
- LungCT global-NCC+BE rows: diagnostic/ablation objectives, not the main
  pTVReg-comparable objective.
- Official pTVReg: local CC plus isoTV, with a linear control grid and
  minFunc/L-BFGS; it is not identical to the repo-local cubic B-spline/Adam
  pTV-style implementation.

Recommended camera-ready rule: use objective labels in every table/caption where
methods cross OASIS/LungCT or pTVReg-style/global-NCC objective families. Do not
compare raw `alpha` values across objective families; compare ARC curves and
common posthoc regularity metrics.

## Sources Checked

Code:

- `scripts/experiments/oasis_alpha_sweep/common.py`
- `scripts/experiments/oasis_alpha_sweep/run_one.py`
- `scripts/experiments/lungct_no_sweep_repro/common.py`
- `scripts/experiments/lungct_no_sweep_repro/run_one.py`
- `external/IDIR/objectives/ncc.py`
- `external/IDIR/objectives/regularizers.py`
- `external/SINR/models/models.py`
- `external/SINR/objectives/regularizers.py`

Experiment/documentation records:

- `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/summary_runs.json`
- `experiments/2026-07-04_oasis_seed3_first100_extension/progress_log.md`
- `experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/`
- `experiments/2026-06-25_lungct_directcp32_global_ncc_be_case01_case08_alpha_sweep/`
- `experiments/2026-06-26_lungct_case08_be_impl_compare_alpha1e4/`
- `experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity/`
- `docs/implementation/idir_sinr_architecture_training.md`
- `docs/implementation/directcp_bspline_optimization.md`
- `docs/reports/protocol_warnings/`
- `docs/offgrid_submission/30_The_Right_Prior_for_the_Rig.pdf`
- `docs/offgrid_submission/review.md`

## Definitions Used in This Audit

### Global NCC

The repo-local `ncc_loss` in both OASIS and LungCT runners is a global negative
Pearson-style NCC over the tensor passed into the function:

```text
cc = mean((x1 - mean(x1)) * (x2 - mean(x2)))
loss = -cc / (std(x1) * std(x2) + eps)
```

There is no local window or Gaussian kernel inside this function. If the caller
passes sampled fixed/moving intensities, the global NCC is computed over those
samples. If the caller passes full volumes, it is computed over the full supplied
volume.

Source anchors:

- OASIS: `scripts/experiments/oasis_alpha_sweep/common.py:ncc_loss`
- LungCT: `scripts/experiments/lungct_no_sweep_repro/common.py:ncc_loss`
- Upstream IDIR: `external/IDIR/objectives/ncc.py:NCC`

### pTV-Style Local CC

The LungCT pTV-style branch implements local CC with an anisotropic Gaussian
kernel whose physical sigma is controlled by `ptv_lcc_sigma_mm`. The current
default is `2.1 mm`. At each stage, sigma in pixels is derived from the current
stage pixel spacing. The loss is an integrated/summed `1 - lcc` map, optionally
multiplied by the pTV-style border mask.

Source anchors:

- `scripts/experiments/lungct_no_sweep_repro/common.py:ptv_lcc_sigma_zyx`
- `scripts/experiments/lungct_no_sweep_repro/common.py:make_ptv_lcc_fixed_cache`
- `scripts/experiments/lungct_no_sweep_repro/common.py:ptv_local_cc_loss_with_cache`
- `scripts/experiments/lungct_no_sweep_repro/common.py:sampled_ptv_local_cc_loss`

### IDIR Coordinate-Autograd BE

IDIR BE differentiates the SIREN displacement with respect to normalized input
coordinates, then differentiates again to form second and mixed derivative
terms. The current code preserves the original IDIR scale convention: after
taking means of the derivative terms, it divides the loss again by
`batch_size`.

This matters because IDIR raw `alpha` values are not numerically comparable to
the spline `alpha` values. The meaningful comparison is the achieved ARC
position and common posthoc regularity metrics, not the raw regularization
weight alone.

Source anchors:

- `scripts/experiments/oasis_alpha_sweep/run_one.py:idir_bending_energy`
- `scripts/experiments/lungct_no_sweep_repro/common.py:idir_bending_energy`
- `external/IDIR/objectives/regularizers.py:compute_bending_energy`
- `docs/implementation/idir_sinr_architecture_training.md`

### Analytic Cubic B-Spline BE

For spline methods, analytic BE is computed from continuous cubic B-spline
derivative fields sampled at voxel centers. The BE terms are the standard
second and mixed derivatives:

```text
BE(u) = mean(u_zz^2 + u_yy^2 + u_xx^2
             + 2*u_zy^2 + 2*u_zx^2 + 2*u_yx^2)
```

The implementation constructs derivative fields by replacing the cubic basis
kernel along an axis with the derivative kernel, applying the expected
`(1 / cps)^q` derivative scaling.

Source anchors:

- OASIS: `scripts/experiments/oasis_alpha_sweep/common.py:analytic_bspline_sampled_be`
- LungCT: `scripts/experiments/lungct_no_sweep_repro/common.py:analytic_bspline_sampled_be`
- Method documentation: `docs/implementation/directcp_bspline_optimization.md`

### Dense Forward-Difference BE and Diffusion

Dense forward-difference BE/diffusion operate on a sampled dense displacement
array. They are used as common posthoc audit metrics and for some explicit
diagnostic training variants. They are not the OASIS main spline training
regularizer.

Source anchors:

- OASIS posthoc/common metric: `scripts/experiments/oasis_alpha_sweep/common.py:forward_valid_be`
- OASIS posthoc/common metric: `scripts/experiments/oasis_alpha_sweep/common.py:forward_valid_diffusion`
- LungCT diagnostic training variant: `scripts/experiments/lungct_no_sweep_repro/common.py:dense_forward_difference_be`
- LungCT posthoc metrics: `scripts/experiments/lungct_no_sweep_repro/common.py:diffusion_loss_vox`
- LungCT posthoc metrics: `scripts/experiments/lungct_no_sweep_repro/common.py:bending_energy_vox`

### Jacobian and Log-Jacobian Diagnostics

Central-difference or `np.gradient` style derivatives are used for Jacobian,
folding, and log-Jacobian diagnostics. They should not be described as the main
BE training loss.

Source anchors:

- OASIS: `scripts/experiments/oasis_alpha_sweep/common.py:jacobian_determinant_zyx`
- LungCT: `scripts/experiments/lungct_no_sweep_repro/common.py:jacobian_determinant_zyx`

## OASIS Current Usage

### Current OASIS Training Objectives

| Method family | Method keys / rows | Similarity used in training | Training domain | Regularizer used in training | Notes |
|---|---|---|---|---|---|
| IDIR / INR-Dense | `idir_original`, `idir_original_dense_siren_be_autograd` | sampled global NCC | sampled coordinates from fixed raw image-positive foreground, `image > 0` | original-style coordinate-autograd IDIR BE | Includes extra divide by coordinate `batch_size`; full-image global NCC can be logged as a diagnostic but does not replace the sampled training loss. |
| SINR / INR-BSCP cps4 | `sinr_cps4_be_bspline_analytic` | full-image global NCC | full OASIS image grid | analytic cubic B-spline BE | SIREN predicts control values; cubic B-spline expansion produces the dense displacement. |
| SINR / INR-BSCP cps2 | `sinr_cps2_be_bspline_analytic` | full-image global NCC | full OASIS image grid | analytic cubic B-spline BE | Same objective family as cps4, different control spacing. |
| D-BSCP cps4 | `direct_cp_cps4_be_bspline_analytic` | full-image global NCC | full OASIS image grid | analytic cubic B-spline BE | Directly optimized control values, no INR generator. |
| D-BSCP cps2 | `direct_cp_cps2_be_bspline_analytic` | full-image global NCC | full OASIS image grid | analytic cubic B-spline BE | Same objective family as cps4, different control spacing. |
| MR-D-BSCP 8-4-2 | `direct_cp_8_4_2_be_bspline_analytic_multistage` | full-image global NCC at each stage image level | current stage image grid | analytic cubic B-spline BE | Direct controls with coarse-to-fine schedule and exact dyadic cubic B-spline knot insertion between stages. |

### OASIS IDIR Details

The IDIR branch samples a mini-batch of foreground coordinates from the fixed
raw image-positive mask, evaluates the SIREN displacement at those coordinates,
warps moving intensities at `coords + disp`, samples fixed intensities at
`coords`, and optimizes:

```text
loss = ncc_loss(fixed_samples, warped_samples)
       + alpha * idir_bending_energy(coords, disp, batch_size)
```

The `summary_runs.json` from the OASIS seed-3 first10 alpha sweep records this
as:

```text
image_loss = idir_global_ncc_sampled
regularizer = be_native_analytic
sample_domain = fixed_image_positive
be_training_reduction =
  original IDIR coordinate-autograd analytic BE;
  sampled spatial means then divided by batch size
```

Interpretation: this is a sampled global NCC objective, not local NCC. The
sampling domain is fixed raw image-positive foreground, not segmentation label
foreground.

### OASIS SINR and D-BSCP Details

The single-stage spline branch expands cubic B-spline controls to a dense field,
warps the full moving image, and optimizes:

```text
loss = ncc_loss(fixed_image_full, warped_full)
       + alpha * analytic_bspline_sampled_be(control_disp, image_shape, cps)
```

This applies to both SINR/INR-BSCP and D-BSCP in the repo-local OASIS runs. The
only architectural difference is that SINR predicts the control values with a
SIREN, while D-BSCP directly stores the control values as trainable parameters.

The OASIS first10 summary records the spline rows as:

```text
image_loss = sinr_ncc
regularizer = be_bspline_analytic
sample_domain = none_full_image_ncc
be_training_reduction =
  analytic cubic B-spline BE sampled at voxel centers;
  spatial mean plus component mean
```

Implementation caveat: the vendored upstream SINR source defaults to an `ncc`
criterion, but its `bending_regularization` path for B-spline transformations
calls a dense first-derivative L2-style regularizer. The repo-local OASIS paper
runner and the completed repo-local summaries should be treated as the source of
truth for these experiments: SINR/INR-BSCP, D-BSCP, and MR-D-BSCP use analytic
cubic B-spline BE in the current paper runs.

### OASIS MR-D-BSCP Details

MR-D-BSCP uses the same global-NCC plus analytic cubic B-spline BE objective
family as the single-stage spline rows, but applies it stage-wise across a
coarse-to-fine control schedule. The paper-facing OASIS schedule is the
`8-4-2` DirectCP multistage row. The code path uses:

```text
data_loss = full-stage-image global NCC
reg_loss = analytic cubic B-spline BE on the current stage control grid
loss = data_loss + alpha * reg_loss
```

The stage transition is exact crop-aligned dyadic cubic B-spline knot insertion.
This should be described as a multiresolution optimization path plus the same
spline BE objective family, not as a different image-similarity loss.

### OASIS Diffusion Clarification

Diffusion appears in OASIS output tables and audit records as a regularity
metric. It is not used as the main OASIS training regularizer for IDIR,
SINR/INR-BSCP, D-BSCP, or MR-D-BSCP.

Use this wording in camera-ready text:

```text
On OASIS, all main continuous-method training rows use global NCC plus BE.
IDIR keeps the original sampled NCC and coordinate-autograd BE convention;
the spline methods use full-image global NCC and analytic cubic B-spline BE.
Diffusion is reported as a posthoc regularity metric.
```

## LungCT Current Usage

### Current LungCT Training Objectives

| Method family | Method keys / rows | Similarity used in training | Training domain | Regularizer used in training | Camera-ready classification |
|---|---|---|---|---|---|
| Default IDIR / INR-Dense | `idir_be_default_lungmask` | sampled global NCC | sampled fixed-phase lung-mask coordinates on the pTV landmark crop / 1 mm grid | IDIR coordinate-autograd BE | IDIR-style baseline. Tag separately from pTVReg-comparable spline rows. |
| IDIR pTV pilot | `idir_sampled_ptv_lcc_tv` | sampled pTV-style local CC | sampled pTV border-masked crop coordinates | sampled analytic physical-unit TV of the SIREN displacement Jacobian | Pilot/variant, not the default IDIR row. |
| SINR cps4 pTV-style | `sinr_cps4_ptv_lcc_isotv` | pTV-style local CC | pTV landmark crop, 5-voxel border mask, no lung-mask loss weighting | physical isoTV on cubic B-spline controls | Main pTV-style spline objective. |
| D-BSCP cps4 pTV-style | `direct_cp_cps4_ptv_lcc_isotv` | pTV-style local CC | pTV landmark crop, 5-voxel border mask, no lung-mask loss weighting | physical isoTV on cubic B-spline controls | Main pTV-style spline objective. |
| MR-D-BSCP pTV-style | `direct_cp_32_16_8_4_ptv_lcc_isotv` and COPD `64-32-16-8-4` records | pTV-style local CC at each image-pyramid stage | pTV landmark crop, stage-aware border mask, no lung-mask loss weighting | physical isoTV on cubic B-spline controls | Main pTVReg-comparable spline objective, with implementation caveats below. |
| Single-stage global NCC + FD-BE | `sinr_cps4_global_ncc_fd_be`, `direct_cp_cps4_global_ncc_fd_be` | full-crop global NCC | full pTV crop, no border/lung-mask loss weighting | dense forward-difference BE | Diagnostic/ablation. |
| MR global NCC + analytic BE | `direct_cp_32_16_8_4_global_ncc_bspline_be` | full-crop global NCC at each stage | full pTV crop, no border/lung-mask loss weighting | analytic cubic B-spline BE | Diagnostic/ablation. |
| MR global NCC + FD-BE | `direct_cp_32_16_8_4_global_ncc_fd_be` | full-crop global NCC at each stage | full pTV crop, no border/lung-mask loss weighting | dense forward-difference BE | Diagnostic/ablation. |

### LungCT Default IDIR Details

The default LungCT IDIR row trains on sampled coordinates from the fixed-phase
lung mask. For 4DCT this is the fixed T00 lung mask. The loss is:

```text
loss = ncc_loss(fixed_samples, warped_samples)
       + alpha_bending * idir_bending_energy(coords, disp, batch_size)
```

The runner may build pTV-LCC caches and full-crop global-NCC diagnostics, but
those evaluations are logged diagnostics and do not change the training loss.

Use this wording:

```text
The default LungCT IDIR reproduction uses IDIR-style sampled global NCC over the
fixed-phase lung-mask coordinate domain plus original-style coordinate-autograd
BE. pTV-style local CC, when enabled for this row, is diagnostic unless the
method key is the explicit sampled-pTV-LCC pilot.
```

### LungCT pTV-Style Spline Details

The pTV-style spline branch uses a pTV landmark crop on a 1 mm registration grid.
For pTV-style training rows, lung masks are not used to weight the training loss;
the fixed lung mask is used for foreground regularity metrics. The image loss is
pTV-style local CC:

```text
sigma = 2.1 mm
loss_map = 1 - local_cc(fixed, warped)
loss = sum(border_mask * loss_map) * voxel_volume
```

The regularizer is physical-unit isoTV on the control values:

```text
reg = sum(sqrt(||grad control_phys||^2 + csqrt)) * control_cell_volume
loss = ptv_lcc_loss + ptv_isotv_weight * reg
```

Current default settings:

```text
ptv_lcc_sigma_mm = 2.1
ptv_isotv_weight = 0.11
ptv_csqrt = 5e-3
border_mask_voxels = 5
optimizer = Adam
```

For multistage MR-D-BSCP, the same objective is evaluated at each stage image
level with stage-aware pixel spacing and domain control spacing. The DirectCP
multistage transfer remains exact dyadic cubic B-spline knot insertion.

### LungCT Global-NCC+BE Details

The LungCT global-NCC+BE rows are deliberately different from the pTV-style
objective:

```text
data_loss = full-crop global NCC
reg_loss = analytic cubic B-spline BE
```

or

```text
data_loss = full-crop global NCC
reg_loss = dense forward-difference BE
```

These rows use no lung-mask or border-mask weighting in the training loss.
They are useful to separate effects of similarity, regularization, and BE
implementation, but they should be labeled as diagnostic/ablation rows, not as
"pTVReg loss" rows.

### Official pTVReg Loss and What "Matched" Means

Official pTVReg lung scripts use:

- local cross-correlation, not global NCC
- LCC sigma `2.1 mm`
- isoTV weight `0.11`
- `csqrt=5e-3`
- no lung-mask optimization in the standard pTV crop setup
- pTV crop / 1 mm preprocessing for the paper-comparable rows
- linear control grid in pTVReg
- minFunc/L-BFGS optimizer

The repo-local pTV-style DirectCP/MR-D-BSCP rows match the named loss family
better than the older global-NCC+BE rows: local CC plus isoTV, with the same
main pTVReg sigma/weight/csqrt/border-mask convention. However, they are still
not a bitwise pTVReg reproduction because they use PyTorch, cubic B-spline
controls, Adam, and exact dyadic knot insertion rather than pTVReg's full
MATLAB optimization stack.

Recommended phrasing:

```text
For LungCT spline methods, "pTVReg-comparable loss" means the pTV-style
local-CC plus isoTV objective family with pTV crop/1 mm preprocessing and
no lung-mask optimization. It does not mean the entire pTVReg implementation is
identical.
```

Avoid this phrasing:

```text
MR-D-BSCP is exactly pTVReg with a different parameterization.
```

Use this instead:

```text
MR-D-BSCP uses a pTV-style local-CC plus isoTV objective, while retaining the
repo-local cubic B-spline parameterization, Adam optimizer, and exact dyadic
control-grid prolongation.
```

## Regularity Clarification for Camera-Ready Text

Use four distinct labels:

1. `IDIR autograd BE`: training regularizer for IDIR rows. It is computed from
   SIREN coordinate Hessians and keeps the original extra divide by batch size.
2. `analytic cubic B-spline BE`: training regularizer for OASIS spline main rows
   and for LungCT global-NCC+analytic-BE diagnostic rows. It differentiates the
   continuous cubic B-spline field.
3. `dense forward-difference BE/diffusion`: common posthoc regularity metrics
   and explicit diagnostic training variants. These operate on sampled dense
   displacement arrays.
4. `Jacobian/log-Jacobian derivatives`: central-difference or `np.gradient`
   style diagnostics for folding, SD-logJ, and related metrics. These are not
   the main BE training loss.

The paper can say that ARC comparisons use common regularity axes, but the
training regularizer is not literally the same implementation for all method
families. That distinction is important because the reviewer explicitly raised
regularization as a possible confound.

## Suggested Frozen Camera-Ready Standard

This section is a recommendation for review, not a committed policy.

### OASIS Main Objective Standard

Use global NCC plus BE for main OASIS comparisons:

- IDIR/INR-Dense keeps original-style sampled global NCC plus coordinate-autograd
  IDIR BE over the fixed `image > 0` coordinate domain.
- SINR/INR-BSCP and D-BSCP use full-image global NCC plus analytic cubic
  B-spline BE.
- MR-D-BSCP uses the same full-image global NCC plus analytic cubic B-spline BE
  at each stage, with exact dyadic cubic B-spline control prolongation.
- Diffusion remains a reported posthoc regularity metric unless a future run is
  explicitly tagged as a diffusion-training ablation.

Recommended label:

```text
OASIS global-NCC+BE standard
```

### LungCT Main Objective Standard

Use pTV-style local CC plus isoTV for main pTVReg-comparable LungCT spline rows:

- SINR/INR-BSCP cps4, D-BSCP cps4, and MR-D-BSCP use pTV-style LCC with
  `sigma=2.1 mm`, 5-voxel border masking, no lung-mask loss weighting, and
  physical control isoTV.
- Default isoTV settings remain `ptv_isotv_weight=0.11` and `ptv_csqrt=5e-3`
  unless an experiment is explicitly an alpha/weight sweep.
- Fixed lung masks are used for foreground regularity metrics, not to weight the
  pTV-style optimization loss.

Recommended label:

```text
LungCT pTV-style LCC+isoTV standard
```

### LungCT IDIR Standard

Keep the default IDIR LungCT row explicitly tagged:

```text
IDIR-style fixed-lung-mask sampled global NCC + IDIR autograd BE
```

If the sampled pTV-LCC + TV IDIR pilot is included, keep it separate from the
default IDIR row:

```text
IDIR sampled pTV-LCC + sampled physical TV pilot
```

This avoids implying that the original IDIR reproduction and pTV-style spline
rows optimize the same objective.

### LungCT Ablation Labeling Standard

Any LungCT result using full-crop global NCC plus analytic B-spline BE or
full-crop global NCC plus dense forward-difference BE should be labeled as:

```text
global-NCC+BE ablation
```

or

```text
diagnostic global-NCC+BE row
```

Do not label those rows as:

```text
pTVReg loss
```

### Alpha and Regularity Standard

Do not compare raw regularization weights across objective families:

- IDIR BE has the original extra divide by `batch_size`.
- Analytic B-spline BE is averaged over sampled voxel-center derivative fields.
- Dense forward-difference BE has a different numerical scale.
- pTV isoTV is an integrated physical-unit TV penalty on control values.

Use ARC curves and common posthoc regularity metrics for cross-method claims.
When discussing training strength, report the objective family and, where useful,
the observed `alpha * reg_loss` scale rather than the raw `alpha` alone.

## Suggested Paper/README Wording

Short Methods wording:

```text
For OASIS, IDIR used the original sampled global-NCC image loss over fixed
image-positive coordinates and the original coordinate-autograd BE regularizer.
Spline-based methods used global NCC over the full image and analytic cubic
B-spline BE sampled at voxel centers. For LungCT, pTVReg-comparable spline rows
used a pTV-style local-CC objective with sigma 2.1 mm, a 5-voxel border mask, and
physical isoTV on control values with weight 0.11 and csqrt 5e-3. Default IDIR
LungCT rows retained the IDIR-style fixed-lung-mask sampled global-NCC plus
autograd-BE objective and are therefore protocol-tagged separately.
```

Caption wording for LungCT MR-D-BSCP:

```text
MR-D-BSCP uses a pTV-style LCC+isoTV objective on the pTV 1 mm landmark crop
(sigma 2.1 mm, border mask 5, isoTV weight 0.11, csqrt 5e-3), while retaining
repo-local cubic B-spline controls, Adam optimization, and exact dyadic knot
insertion.
```

Reviewer-response wording:

```text
We clarified that the regularizer is not a single identical implementation
across all parameterizations. IDIR uses the original coordinate-autograd BE
scale convention; OASIS spline methods use analytic cubic B-spline BE; LungCT
pTVReg-comparable spline methods use pTV-style isoTV; and dense finite-difference
BE/diffusion are reported as common audit metrics or used only in explicitly
labeled diagnostic variants.
```

## Remaining Review Decisions

- Decide whether camera-ready text should call the LungCT spline objective
  "pTVReg loss" only in shorthand after a precise first definition, or avoid
  that phrase entirely in favor of "pTV-style LCC+isoTV".
- Decide whether to include the IDIR sampled pTV-LCC + TV pilot in the main
  release tables or only in supplemental diagnostics.
- Decide whether to add a small methods table to the paper listing, for each
  dataset/method family, the similarity loss, regularizer, optimization domain,
  and alpha/weight grid.
- If code-level standardization follows this report, add assertions or summary
  metadata checks so future runs cannot silently mix objective labels.
