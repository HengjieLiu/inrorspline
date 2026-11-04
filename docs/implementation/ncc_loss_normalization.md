---
doc_type: implementation_note
status: audited_current_formulas
audit_date: 2026-10-12
---

# NCC loss definitions and normalization

The current squared local NCC, signed global NCC, and squared global NCC use
compatible population-moment correlation normalization. They do not have a
calibrated, equal similarity-to-diffusion balance at equal lambda. Squaring and
local versus global support change the optimization problem even when the
variance convention and regularizer are unchanged.

FeDDA currently has completed signed GNCC and squared LNCC training sweeps.
Squared GNCC also exists in the direct dense registration runner. The FeDDA
squared-GNCC adapter passed preflight and its four-run sweep was launched on
2026-10-12; results remain pending. It retains the normalization below and
recomputes statistics for every image and training direction.

## Three formulas

Let F be the fixed image and W the warped moving image. For support S, define
population means, covariance, and variances:

\[
\mu_F=\operatorname{mean}_S F,\quad
\mu_W=\operatorname{mean}_S W,\quad
c_S=\operatorname{mean}_S[(F-\mu_F)(W-\mu_W)],
\]

\[
v_F=\operatorname{mean}_S[(F-\mu_F)^2],\quad
v_W=\operatorname{mean}_S[(W-\mu_W)^2],\quad
\rho_S=\frac{c_S}{\sqrt{\max(v_Fv_W,\epsilon)}}.
\]

| Objective | Correlation support | Similarity loss | Sign behavior |
| --- | --- | --- | --- |
| `1 - LNCC^2` | Separate 9 x 9 x 9 window at every voxel p | `1 - mean_p(rho_p**2)` | Rewards either correlation sign |
| `1 - GNCC` | Entire image, independently per batch element b | `mean_b(1 - rho_b)` | Penalizes negative correlation |
| `1 - GNCC^2` | Entire image, independently per batch element b | `mean_b(1 - rho_b**2)` | Rewards either correlation sign |

LNCC squares each local correlation before averaging. Squared GNCC must square
each image's correlation before averaging over the batch; neither is a square
of an averaged correlation. The ideal ranges are [0, 2] for signed GNCC and
[0, 1] for the squared objectives, up to numerical error.

## Current normalization and stabilization

- Moments use means, not sums or sample standard deviations. The current GNCC
  does not use the older mixed mean-covariance/sample-standard-deviation formula
  or its custom stabilized backward operation. It uses ordinary autograd.
- FP32 training uses `epsilon = 1.1920928955078125e-7`. Native LNCC obtains it
  from `torch.finfo(dtype).eps`; current GNCC explicitly records that FP32
  constant. These match for this FP32 protocol, not necessarily arbitrary dtypes.
- GNCC divides covariance by `sqrt(clamp_min(v_F * v_W, epsilon))`.
  LNCC squared divides squared covariance by
  `clamp_min(v_F * v_W, epsilon)`. Squaring the stabilized global correlation
  therefore gives the same denominator convention. This is not an additive
  epsilon and not independent clamping of both variances.
- LNCC computes moments with `E[F*W] - E[F]*E[W]` and analogous variances;
  GNCC explicitly centers the images. These are mathematically equivalent
  population conventions, but their floating-point cancellation can differ.
- LNCC uses a uniform normalized 9-cubed filter, stride 1, zero padding 4,
  and averages over every window center. Boundary windows retain the full
  729-element divisor, including the padded zeros.
- GNCC uses the whole image, including background, without a local-window
  padding operation. Neither loss uses a foreground training mask.
- FeDDA retains native stored intensities and LIA-to-RAS preprocessing. There
  is no extra image z-scoring, initial-loss normalization, gradient calibration,
  or multiplicative similarity scaling. The similarity multiplier is 1.

Without the floor, correlation is normalized by the local or global standard
deviations. The floor weakens that scale invariance in low-variance regions;
zero padding also affects local boundary statistics. Constant blank local
windows have zero covariance and contribute loss 1, generally without useful
alignment gradients. Averaging those windows can dilute active-region
gradients. Global and local normalization therefore weight anatomy and
background differently despite using the same epsilon.

The optional native `NCC_vfa(keep_sign=True)` is not signed Pearson NCC: it
multiplies an already squared score by the covariance sign. It is not any of
the three objectives in this comparison.

## Shared diffusion and the meaning of lambda

All three FeDDA objectives use the proposed/common total loss

\[
L=L_{\rm sim}+\lambda R(u),\qquad
R(u)=\frac{1}{3}\sum_{a\in\{x,y,z\}}
\operatorname{mean}_{\text{valid edges, components}}[(\Delta_a u)^2].
\]

This is native `Grad3d(penalty='l2')`, with no loss multiplier: unmasked
forward-valid differences of the total voxel-unit displacement, component
means, then the mean over the three axes. It is not bending energy, a
normalized-coordinate penalty, or a penalty on an incremental field.

Foreground diffusion in reports restricts each axis to edges whose two
endpoints satisfy `fixed_raw_image > 0`. It is a posthoc diagnostic, not exactly
the unmasked training regularizer. Preserve full-volume diffusion alongside it.

Even on identical global support,

\[
1-\rho^2=(1+\rho)(1-\rho),\qquad
\nabla_u(1-\rho^2)=2\rho\,\nabla_u(1-\rho).
\]

At rho = 0.9, signed loss is 0.10 and squared loss is 0.19; the squared
similarity gradient is 1.8 times the signed gradient. A local heuristic for
matching the data-to-regularizer gradient ratio is
`lambda_squared approximately 2*rho*lambda_signed`, not its inverse.
This is not a fixed conversion: rho changes during training and between cases,
and Adam's optimization path is not guaranteed to match after rescaling.
Near zero correlation, the squared gradient vanishes; at negative correlation
its direction reverses relative to signed NCC.

There is no universal scalar conversion between LNCC and GNCC either. Compare
lambda sweeps and achieved regularity, rather than interpreting equal lambda
as an equal effective deformation constraint. Do not silently insert a factor
of one half into squared GNCC; that would be an explicitly rescaled variant.

## What the FeDDA comparison can establish

Signed GNCC versus squared GNCC holds global support fixed and changes
squaring. Squared GNCC versus squared LNCC holds the power fixed and changes
local versus global statistics, including padding and low-variance behavior.
The existing signed-GNCC versus squared-LNCC comparison changes both factors.
The three curves help separate these factors but are not a full factorial
experiment because signed local NCC is not included.

The approved follow-up is
[P58 FeDDA squared global NCC training](../../experiments/2026-10-12_oasis_fedda_gncc2_training/codex_execution_spec.md).
Single-seed training and a previously inspected test cohort limit causal and
generalization claims. Across-case SD is not training-seed uncertainty.

## Code references

- [FeDDA squared GNCC adapter](../../scripts/experiments/oasis_alpha_sweep/fedda_gncc2_training.py):
  `GlobalNCCSquared.correlation` and `__call__`; detached rho diagnostics do not
  cache image moments or change gradients.
- [Native FeDDA setup and directional updates](../../scripts/experiments/oasis_alpha_sweep/fedda_training.py):
  `setup_native`, `directional_loss`, and `train_pair`.
- [Current FeDDA GNCC adapter](../../scripts/experiments/oasis_alpha_sweep/fedda_gncc_training.py):
  reuses the stateless `GlobalNCC` below and the native regularizer.
- [Population GNCC implementation](../../scripts/experiments/oasis_alpha_sweep/sitreg_gncc_training.py):
  `GlobalNCC.__call__` and `PROTOCOL`.
- [Single-pair signed and squared GNCC](../../scripts/experiments/oasis_alpha_sweep/gncc_dense.py):
  `GNCC.correlation` and `GNCC.__call__`. This class caches fixed-image moments
  for one registration pair;
  do not reuse it unchanged for minibatch DL training.
- `${WAVEREG_ROOT}/code/utils_losses_ncc.py`: `NCC_vfa.forward`.
- `${WAVEREG_ROOT}/TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph/losses.py`:
  `Grad3d.forward`.

The planning log records the audited working-tree source hashes. Historical
sources and accepted experiment outputs must remain unchanged.
## Checkpoint recovery and subsequent squared-GNCC SITReg sweep

The approved 2026-10-13 recovery leaves the loss definitions below unchanged.
SITReg's new `sitreg_gncc2_training` adapter differs from the signed adapter
only in loss identity and `mean(1-rho²)` replacing `mean(1-rho)`. The native
joint bidirectional reduction, total voxel-field diffusion, optimizer, data
pairing and validation checkpoint selection are retained.

Native training keeps a shuffled DataLoader iterator across epochs. Restoring
only a checkpoint RNG and making a fresh loader is not equivalent: a partially
consumed sampler permutation would be lost. Recovery replays the native
metadata-only pairing dataset and validation seed draws from the seed42 model
initialization, verifies every completed epoch's pair hash and exact CPU/
Python/NumPy RNG states, then retains the live iterator and restores saved
CUDA RNG and Adam/AMSGrad state. The historical CUDA backend remains non-bitwise
reproducible. See P59 for frozen-source provenance and failed-artifact retention.
