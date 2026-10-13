# DL-matched NCC9/diffusion registration

This is the code-derived specification for experiment P43, not a change to
historical MR-D-BSCP. Sources:

- `scripts/experiments/oasis_alpha_sweep/dl_ncc_diffusion.py`: geometry,
  objective, stage schedules and optimizer.
- `scripts/experiments/oasis_alpha_sweep/run_oasis_dl_ncc_diffusion_one.py`:
  input validation, prefix/branch initialization, serialization and metrics.
- `scripts/experiments/oasis_alpha_sweep/launch_oasis_dl_ncc_diffusion_sweep.py`:
  dependency scheduling, smoke gates and provenance checks.
- `scripts/reports/oasis_dl_ncc_diffusion_first10_arc.py`: complete-cohort
  validation, canonical reference comparison and ARC-style plots.
- `tests/test_oasis_dl_ncc_diffusion.py`: mathematical and interface tests.

## Representations and coordinates

All optimized parameters are normalized-grid XYZ displacements, with
align_corners=True. The three representations are cubic cps8/4/2, cubic
cps8/4/2/1, and cubic cps8/4/2 followed by a directly optimized dense field.
The final dense field is the detached exact clone of the expanded cps2 field;
there is no final control lattice, residual composition, or second warp.

The full-resolution path uses the established separable cubic expansion and
crop. The pyramid path evaluates that continuous cubic basis directly at the
endpoint-aligned native coordinates corresponding to the resized image grid.
It does not use the historical integer domain-CPS shortcut, interpolate the
controls, or resample a previously expanded dense field. Between spline stages,
dyadic knot insertion preserves the continuous field up to FP32 roundoff.

## Objective and units

Images retain their stored float32 intensities after the standard orientation
transform. Pyramid images use the existing Gaussian antialiasing plus
align_corners=True trilinear resizing. Image sampling uses border padding;
canonical nearest-neighbor segmentation evaluation separately retains zero
padding.

For normalized displacement q on a stage grid of shape D,H,W, set

    u = q * [(W-1)/2, (H-1)/2, (D-1)/2]
    L = NCC9(fixed, moving(identity + q)) + lambda * R(u)
    R(u) = (mean(dx(u)^2) + mean(dy(u)^2) + mean(dz(u)^2)) / 3

Each diffusion mean includes all valid neighboring voxels and all three
components. It is unmasked and applies to the total expanded displacement,
not to controls or a refinement residual. Conversion to voxel units remains
inside autograd. Lambda is identical at every stage, with no rescaling or
initial-loss normalization.

NCC9 matches the audited DL training NCC_vfa implementation: 9x9x9 windows,
zero-padded local sums divided by729, squared covariance divided by the
variance product clamped below by FP32 epsilon, and loss1-mean(correlation).
It is squared local NCC, not signed global NCC or the signed feature NCC from
earlier experiments. Only frozen fixed-image moments are cached. There is no
segmentation loss, VFA feature loss, or displacement supervision.

## Optimization and checkpoint semantics

Every stage starts fresh Adam with betas0.9/0.999, epsilon1e-8, no weight decay
and no AMSGrad. Learning rates are0.03/0.01/0.003/0.001 in normalized units.
Full budgets and stopping settings are in PROTOCOL and the experiment spec.
Each history entry is the evaluated state after exactly the recorded number
of updates. Initialization is state0 and is eligible as the best checkpoint.

The lowest total-loss state is saved and handed to the next stage, while the
terminal state and its Adam state are also saved. Losses are compared only
within a stage, never across evaluation scales. Prefix stages stop on the
specified median-window plateau; both final refinement types allow two LR
halvings before stopping on the next plateau. Selection never uses label
metrics or an epoch chosen from a registration-metric curve.

FP32, seed1, cuDNN benchmark=False/deterministic=True, and disabled TF32 are
explicit. CUDA grid_sample backward can still be nondeterministic; these
settings do not imply global bitwise reproducibility.

## Experiment and reporting boundaries

Ten seed3-shuffled cases, seven lambdas, three representations and two loss
resolution modes yield420 outputs. Sharing is limited to140 fresh prefixes
with identical case, lambda, resolution, inputs and numerical source hashes.
Historical prefix caches and old saved refinements are not reused.

Original VFA/SITReg fields and accepted reference evaluations are read-only.
All420 successful finite fields and their provenance must pass validation
before final aggregation. Reports include canonical Dice/HD95/symmetric ASD,
foreground SD(log J), folding, diffusion, exclusions, stage diagnostics and
paired reference differences. ARC plots use marginal +/-1 sample-SD ellipses
across ten cases, not confidence ellipses or confidence intervals.

Pyramid mode intentionally changes NCC support and voxel units with evaluation
resolution. Full-resolution mode matches the audited per-direction DL loss,
not the learned model, bidirectional training or network prior. All ten cases
are exploratory tuning. Pipeline compute budgets differ, and competing GPU
workloads make runtime diagnostic rather than a controlled speed comparison.

## Rank4 cps16 extension (P47)

The isolated `cps16_ncc.py` protocol and rank004 worker reuse the unchanged
objective, aligned basis and stage optimizer. They prepend cps16 at LR.03 to
the existing four-stage schedule, using pyramid factors8/4/2/1/1 or all1.
Dense refinement is now index4; its LR reductions follow representation,
not the historical index3. Native fields and canonical label/regularity metrics
are saved after every best-loss stage for posthoc diagnosis only. Sequential
GPU3 execution and scope are fixed in the approved
[spec](../../experiments/2026-10-09_oasis_rank004_cps16_ncc/codex_execution_spec.md).
