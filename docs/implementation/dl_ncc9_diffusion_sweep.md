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

## Fixed-prefix final-stage lambda extension (P60)

The isolated `fixed_prefix_dense.py` runner retains the historical objective,
stage optimizer and canonical evaluation functions. Unlike P43's tied lambda,
it validates a fixed `[1,1,1]` prefix independently of dense lambda and names
each candidate with `[1,1,1,dense_alpha]`. Every final-stage fit is fresh,
including lambda1, and starts from the same saved per-case normalized dense
field with exact cloning. Historical FitSpec validation and sources remain
unchanged. The dense-stage alpha changes only the total-field diffusion weight;
image preprocessing, local NCC9, parameter units, LR and stopping stay frozen.
See [P60 spec](../../experiments/2026-10-13_oasis_fixed_prefix_dense/codex_execution_spec.md).

## Fixed cps8/cps4 prefix to cps2 lambda extension (P61)

P61's isolated `fixed_prefix_cps2.py` branches earlier than P60: it loads
the saved lambda1 cps4 best controls, prolongs to cps2, and freshly optimizes
only cps2 at lambda0.1/0.2/0.5/1/2/5. Stage weights are [1,1,cps2_alpha];
there is no dense stage. Cps2 retains LR0.003, 1300/130 epochs, patience50
and no LR reductions. Historical842 lambda1 is a reproducibility/reference
row, not the initialization. See
[P61 spec](../../experiments/2026-10-13_oasis_fixed_prefix_cps2/codex_execution_spec.md).

## Cps2 lambda5 prefix to dense extension (P62)

P62's `cps2lambda5_dense.py` initializes dense refinement from P61's accepted
cps2lambda5 outputs, preserving stage vectors[1,1,5,dense_alpha]. Its dense
objective and schedule are identical to P60, with an additional requested
lambda10. It validates P61's separate-weight metadata directly, rather than
using the legacy tied-lambda loader. Previous[1,1,1] dense results are read-only
paired initialization comparisons, not reproduction targets. See
[P62 spec](../../experiments/2026-10-13_oasis_cps2lambda5_dense/codex_execution_spec.md).

## Rank0 cps1 comparison after [1,1,5] prefix (P63)

P63's `cps2lambda5_cps1.py` prolongs the identical accepted P61 cps2 controls
to cps1 instead of cloning the dense field as in P62. It retains the exact
full-resolution NCC9 and finite-difference diffusion objective, final lambda1,
normalized LR0.001, fresh Adam and2000/300 adaptive schedule. The regularizer
is evaluated on the expanded total field; it is not analytic B-spline diffusion.
Native-grid prolongation is verified against the saved prefix before training.

The rank0 comparison matches dense accuracy but does not reduce folding.
Equal nominal optimizer settings do not imply identical field-space updates,
and a cps1 cubic representation is not a positive-Jacobian constraint. See
[P63 spec](../../experiments/2026-10-13_oasis_cps2lambda5_cps1/codex_execution_spec.md).

## Bounded retained composition (P64)

`bounded_composed_bspline.py` defines one identity-initialized bounded
increment per cps8/4/2/1. `run_oasis_bounded_composed.py` is the isolated
rank0 worker/launcher. Historical runners and outputs are unchanged.

Each bounded control grid expands once per objective evaluation to native
samples; the elementary map uses their border-trilinear interpolant. Earlier
elementary samples are frozen, but grid sampling still propagates coordinate
gradients. With maps stored oldest-first, evaluation applies them in reverse:
Phi_new=Phi_previous o delta_new. The accumulated map is not repeatedly
resampled. Stage metrics/exports are samples only, never initialization caches.
There is no interstage knot insertion or single final cps1 representation.

For unit control-grid coefficients, the FP64 Lipschitz constant is the maximum
over all native sampling phases, eight cell corners and eight derivative sign
patterns of the L1 norm of the signed sum of derivative stencils. Its inverse
is the coefficient bound. Endpoint-aligned sampling yields control-unit bounds
0.400880185/0.392638037/0.36/0.36 for cps8/4/2/1, distinct from SITReg's
even-stride half-voxel sampling bounds. The independent calculation reproduces
the native SITReg algorithm/lookup when its own phases are used.

Let a=0.99*cps*bound/native_voxel_scale_xyz. Optimize normalized latent p,
with bounded coefficients c=a*tanh(p/a). The derivative dc/dp is1 at zero;
directly applying historical normalized LRs to dimensionless tanh logits would
not preserve this convention. Coefficient saturation and derivative bounds are
recorded, and previous maps are never added as raw displacement increments.

The unchanged loss formula uses the total composed displacement in the current
evaluation grid's voxel units. This changes the deformation representation and
coarse-grid field sampling, not the image pyramid or NCC9/diffusion formula.
Full native-cell corner audits verify elementary contraction.100k same-point
FP64 probes separately evaluate the retained chain and reinterpolated export.
Zero canonical folding or zero sampled off-grid failures is not a universal
claim about every downstream interpolant. See
[P64 spec](../../experiments/2026-10-13_oasis_bounded_composed/codex_execution_spec.md).

## Bounded cps16 extension (P65)

`bounded_composed_cps16.py` and `run_oasis_bounded_composed_cps16.py` add a
separate cps16/8/4/2/1 retained chain. The P64 numerical sources remain
unchanged; the new adapter reuses their composition, scaled tanh, optimizer
and topology helpers. Explicit five-stage metadata fixes final LR reductions
to cps1 rather than historical index3. Weights are[1,1,1,5,1], pyramid factors
8/4/2/1/1, normalized LRs0.03/0.03/0.01/0.003/0.001. New cps16 controls are
14x18x16x3; downstream shapes and schedules retain P64's values.

Endpoint-aligned cps16 coefficient bound is approximately0.4019072194 in
control-grid units, or6.366210355 native voxels/component after0.99 margin.
SITReg's half-voxel counterpart0.4017532975 is tested independently, not
substituted. The loose cumulative per-axis motion ceiling rises5.7990 to
12.1652 voxels. This adds motion capacity, optimizer work and coarser NCC
support, so any improvement is not a causal isolation of saturation.
See [P65 spec](../../experiments/2026-10-13_oasis_bounded_composed_cps16/codex_execution_spec.md).

## Selective bounding (P66)

`selective_bounded_composed.py` and `run_oasis_selective_bounded.py` retain
P64's4-stage composition, schedule and NCC9/diffusion, but bound only the last
1,2 or3 increments. Unbounded controls are exactly the normalized parameter
c=p; bounded controls retain c=a*tanh(p/a). Serialization uses the same
stage-specific transform as optimization. Unbounded applied bounds and
saturation metadata are null rather than implying an unused cap was enforced.

Each fit starts independently at identity; stages are NOT the historical
prolonged-prefix scheme. Fully bounded P64 is the controlled reference;
historical prolonged unbounded/dense are contextual. Bound checks gate only
bounded elementary maps; folded earlier maps or total compositions are
measured and retained, not rejected or repaired. See
[P66 spec](../../experiments/2026-10-13_oasis_selective_bounded/codex_execution_spec.md).

P67 extends these same four bounding masks (including P64 all4) to seed3
first10 without changing the numerical core. The separate first-ten adapter
reuses four immutable rank0 outputs and runs36 identity-initialized fits.
All four variants of a new case share a physical GPU, not an optimized prefix.
Case-level sample SD uses ddof1, with ranks1–9 separated from exploratory
rank0. See [P67 spec](../../experiments/2026-10-13_oasis_selective_bounded_first10/codex_execution_spec.md).

P68 compares two bounded increments per level against two optimizer blocks on
one bounded map. Both use eight blocks over four levels, with the same
per-level lambda/LR/stopping schedule repeated independently. The new-map arm
appends a zero-initialized map; continuation restores the selected same-map
latent, resets Adam/LR/stopping history and replaces that map. Thus the final
retained checkpoint indices are[0..7] versus[1,3,5,7]. Bounds and total-field
diffusion are unchanged, not doubled or divided by two. The new optimizer
only adds explicit initial-latent support; historical source modules remain
immutable. See [P68 spec](../../experiments/2026-10-13_oasis_bounded_two_blocks/codex_execution_spec.md).

P69 extends only the two-increment arm to first10 at stage weights[1,1,5,1]
and[1,1,1,1]. Each level's weight applies unchanged to both increments and
regularizes the total composed field, not the increment. Only cps2 changes;
the first four optimization blocks must agree across vectors. Two pilot
outputs are reused read-only, with18 fresh registrations. A scoped cohort
adapter reuses P68's immutable numerical worker and audit; it does not alter
historical constants on disk. See [P69 spec](../../experiments/2026-10-13_oasis_two_increments_first10/codex_execution_spec.md).

## Rank4 cps16 extension (P47)

The isolated `cps16_ncc.py` protocol and rank004 worker reuse the unchanged
objective, aligned basis and stage optimizer. They prepend cps16 at LR.03 to
the existing four-stage schedule, using pyramid factors8/4/2/1/1 or all1.
Dense refinement is now index4; its LR reductions follow representation,
not the historical index3. Native fields and canonical label/regularity metrics
are saved after every best-loss stage for posthoc diagnosis only. Sequential
GPU3 execution and scope are fixed in the approved
[spec](../../experiments/2026-10-09_oasis_rank004_cps16_ncc/codex_execution_spec.md).
