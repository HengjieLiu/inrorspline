---
doc_type: implementation_summary
status: active_draft
---

# IDIR And SINR Architecture And Training

This note summarizes the implementation behavior used by the current repo-local
experiments. Here, `IDIR` means the original-style dense coordinate-network
method, and `SINR` means the spline-enhanced INR method where a SIREN predicts
cubic B-spline control displacements.

## IDIR

IDIR represents the displacement field directly as a coordinate network:

```text
u_theta: normalized coordinate (x, y, z) -> normalized displacement (dx, dy, dz)
phi(x) = x + u_theta(x)
```

The current runners use a SIREN MLP with default layers:

```text
[3, 256, 256, 256, 3]
```

The activation is sine, the OASIS/LungCT runner default is `omega=32`, and
parameters are optimized with Adam. The output displacement is in the normalized
grid coordinate system used by PyTorch `grid_sample`.

Training samples coordinates from the active foreground domain, evaluates the
moving image at `x + u_theta(x)`, samples the fixed image at `x`, and optimizes:

```text
loss = image_loss + alpha * reg_loss
```

The IDIR image loss in the current repo-local runs is sampled NCC. The
regularizer is coordinate-autograd bending energy computed from second
derivatives of the SIREN displacement with respect to normalized input
coordinates.

Important scale convention:

- The original-style IDIR BE first averages second-derivative terms over the
  sampled coordinates.
- It then divides the result again by the coordinate batch size.
- With the default batch size of `10000`, this downscales the training BE by a
  factor of `10000` relative to the pre-divide mean.

Because of this, raw IDIR alpha values are not numerically comparable with
SINR/DirectCP alpha values. Cross-method regularity comparisons should use
common posthoc metrics, and training-strength discussion should mention
`alpha * reg_loss` rather than raw alpha alone.

For reporting, IDIR native analytic regularity means:

- BE: coordinate-autograd Hessian terms sampled at voxel centers.
- diffusion: coordinate-autograd Jacobian terms sampled at voxel centers.

## SINR

SINR keeps the coordinate-network component but moves the deformation through a
cubic B-spline control lattice. In the repo-local runners:

```text
g_theta: control-point coordinate -> control displacement
u(x) = cubic_bspline_expand(g_theta(control_coords))(x)
phi(x) = x + u(x)
```

The SIREN control generator uses the same basic hidden-width convention as IDIR
unless overridden by a runner:

```text
input_dim = 3
hidden_dim = 256
hidden_layers = 3
output_dim = 3
omega = 50
```

The key architectural distinction from DirectCP is where the control values
come from:

- SINR predicts control displacements by applying a SIREN to control-point
  coordinates.
- DirectCP stores the control displacements themselves as trainable parameters.

Both SINR and DirectCP then use cubic B-spline expansion to produce the dense
deformation field. This shared spline/local-control structure is the reason
DirectCP is a critical control for isolating the INR contribution.

For reporting, SINR native analytic regularity means cubic B-spline analytic
derivatives sampled at voxel centers:

- diffusion from first derivatives
- BE from second and mixed derivatives

## MR-SINR 8/4/2 Revision Variants

The OASIS MR-SINR revision adds two multiresolution SINR variants in
`scripts/experiments/oasis_alpha_sweep/run_one.py`.

`mr_sinr842_composed` uses three independent residual SINR stages at BSCP
`8,4,2`. Stage 8 registers the original moving image. Stage 4 registers the
moving image already warped by the frozen stage-8 pull field. Stage 2 registers
the moving image already warped by the frozen composed stage-8/stage-4 field.
The final deformation is evaluated by pull-field composition:

```text
phi842(x) = phi8(phi4(phi2(x)))
```

The composed final deformation is not a single cubic B-spline field, so final
regularity is reported with dense posthoc metrics. Per-stage training still
uses full-image global NCC plus analytic cubic B-spline BE on the residual
stage controls.

`mr_sinr842_requery` uses one continued SINR generator across BSCP `8,4,2`.
At each stage, the same generator is re-queried at the current control lattice
and optimized as an absolute field. It does not compose fields between stages
and is not exact knot insertion; transition-jump diagnostics record the field
change caused by re-querying at the next lattice.

## OASIS Brain

Primary OASIS protocol:

- Dataset handle: OASIS BrainMR `pair_test_200`.
- Subset: seed-3 shuffled first10 by default.
- Fixed/moving direction: as listed in the seed-3 pair file.
- Foreground rule: fixed raw image-positive mask, `image > 0`.
- Segmentation masks: Dice, HD95, ASD, and visual comparison only.

Current OASIS method setup:

- IDIR original:
  - sampled coordinates from fixed raw image-positive foreground
  - sampled NCC image loss
  - coordinate-autograd BE training regularizer
  - `lr=1e-4`, `epochs=2500`, seed `1`
  - alpha grid `0,0.01,0.1,1,10,100`
- SINR cps4:
  - SIREN-generated cubic B-spline controls with `cps=4`
  - full-image/global NCC training loss
  - analytic cubic B-spline BE training regularizer
  - `lr=1e-4`, `epochs=2500`, seed `1`
  - spline alpha grid `0,10,100,1000,10000,100000`
- SINR cps2:
  - same as SINR cps4 except `cps=2`

OASIS accuracy metrics include Dice, HD95, and ASD. Regularity metrics include
native analytic BE/diffusion, dense forward-difference BE/diffusion,
log-Jacobian spread, folding percentage, and NDV. Foreground-scoped regularity
uses fixed raw image-positive foreground.

## LungCT / 4DCT

The LungCT code supports multiple protocol domains. The paper-facing rows must
keep these tags explicit because native IDIR-style runs and pTV-preprocessed
SINR/DirectCP runs are not identical optimization problems.

Current LungCT IDIR setup:

- Fixed phase: T00.
- Moving phase: T50.
- Training coordinates: fixed T00 lung mask.
- Image loss: sampled NCC.
- Regularizer: IDIR coordinate-autograd BE.
- Default training: `lr=1e-4`, `epochs=2500`, `alpha_bending=10`, seed `1`.

Pilot LungCT sampled pTV-LCC + TV IDIR variant:

- Method key: `idir_sampled_ptv_lcc_tv`.
- Image domain: regenerated pTVReg-style landmark crop, 1 mm registration grid.
- Training coordinates: pTV 1mm border-masked crop samples, not lung-mask
  samples.
- Image loss: sampled pTV-style local CC from Gaussian-weighted integer patch
  offsets around sampled centers.
- Regularizer: sampled analytic physical-unit TV of the SIREN displacement
  Jacobian. This is an IDIR-specific analogue of pTV isoTV because IDIR has no
  control lattice.
- Pilot default: `ptv_isotv_weight=0.11`, `lr=1e-4`, `epochs=2500`, seed `1`.

Current pTV-prep SINR cps4 setup:

- Image domain: regenerated pTVReg-style landmark crop, 1 mm registration grid.
- Control model: SIREN-generated cubic B-spline controls with `cps=4`.
- Image loss: pTV-style local CC.
- Regularizer: pTV-style isoTV on controls.
- Default training: `lr=1e-4`, `epochs=2500`, `ptv_isotv_weight=0.11`, seed
  `1`.
- Border-mask loss weighting is used for pTV loss methods; lung masks are used
  for foreground metrics, not as the pTV loss mask.

Controlled DUAL-INR pTV 1 mm fixed-lung-mask revision:

- Implementation path:
  `scripts/experiments/lungct_dual_inr_controlled/`.
- Protocol `PL` uses the canonical pTVReg-style 1 mm fixed/moving images and
  fixed T00 lung mask, but unlike protocol `P`, it uses that fixed T00 mask as
  the optimization sampling domain as well as the metric mask.
- Pilot variants `PL1D` and `PL1B` keep the released DUAL architecture and
  multiresolution image factors `8,4,2,1` with stage lengths
  `400,400,800,900`.
- `PL1D` uses the released DUAL native spatial regularization branch:
  sampled global NCC plus weighted Laplacian and negative-Jacobian penalties,
  with semantic/lobe/segmentation loss disabled.
- `PL1B` replaces the DUAL native spatial regularizers with IDIR-style
  coordinate-autograd bending energy on the total DUAL displacement. It
  preserves the historical IDIR batch-size scaling and uses default pilot
  weight `10.0`.

LungCT metrics include landmark TRE and deformation regularity. The headline
pTVReg-comparable TRE convention is original/native-crop snap TRE when those
rows are compared to pTVReg-style references. Cropped 1 mm snap/subvoxel TRE
are diagnostics and should be labeled as such.
