# Fourier-input IDIR: controlled first10 bandwidth/diffusion sweep

Approved in conversation on 2026-10-10. Execute on physical GPUs 0,1,2,3,
one new worker per GPU; preserve all existing workloads and prior outputs.
Use the existing artifact-store container, UID:GID 1009:1020, recorded image
tag/digest and current GPU memory/utilization. No commits or raw-data writes.

## Fixed numerical protocol

- Canonical OASIS seed3 shuffled first10 pair file, fixed/moving direction and
  raw-image-positive foreground. Dataset manifest and all inputs are hashed.
- Preserve P53 pair-minmax images, zero-padded trilinear sampling,
  align_corners=True, normalized XYZ network inputs and displacements.
- Fresh Adam LR 1e-4, betas 0.9/0.999, epsilon 1e-8, no weight decay,
  2500 updates, no scheduler/early stopping, terminal checkpoint selection.
- Exactly 10,000 foreground coordinates per update, sampled without replacement
  with torch.randperm and refreshed each update. A dedicated CUDA seed1 stream
  must reproduce P53's full coordinate-sequence hash for each case, regardless
  of architecture, Fourier bandwidth or initialization seed.
- Loss is signed sampled global 1-NCC plus lambda times sampled analytic
  voxel-coordinate displacement diffusion. It is NOT local NCC9 or dense FD.
  Differentiate through the Fourier encoding with respect to original XYZ.
  Preserve component/axis voxel scaling and no extra batch-size division.
- FP32, cuDNN benchmark=False, deterministic=True, TF32 off. CUDA sampling
  backward may remain nondeterministic. Runtime is shared-load diagnostic.

## Encoding and controls

For q in [-1,1]^3, v=(q+1)/2 and
gamma(q)=[q, sin(2*pi*sigma*B0*v), cos(2*pi*sigma*B0*v)].
B0 has 64 frozen standard-Gaussian 3-vectors. Use a dedicated CPU generator
seed1 and save/hash the matrix; do not train B0 or change feature amplitudes.
Draw a fixed 128-row bank and use its prefix so optional feature-count tests
share frequency vectors. The matrix is shared across cases/bandwidths.

- Fourier arm: input131 -> 256 -> 256 -> 256 -> 3, omega32, 166147 parameters.
- Native control: input3 -> 256 -> 256 -> 256 -> 3, 133379 parameters.
- Capacity control: input3 -> 286 -> 286 -> 286 -> 3, 166169 parameters.
- Preserve the existing native SIREN initialization rule, including biases,
  with first-layer weight bounds recomputed as +/-1/input_dim. No unrecorded
  variance matching, output zeroing, gain normalization or initialization tuning.
- Record first pre-sine activation spread, initial voxel displacement,
  similarity/diffusion, their parameter-gradient norms and weighted ratio.
  Compute these on a saved initial-model copy on CPU AFTER each worker's
  optimizer loop, so diagnostic autograd does not perturb the native training
  path. Review smoke diagnostics before releasing the full production sweep.

## Search and selection

Initial grid: sigma 0.5/1/2/4/8/16, lambda 0.3/1/3, first10 = 180 new fits.
Native control adds lambda 0.3/3 = 20 fits; reuse only accepted P53 lambda1
omega32 first10 (read-only). Capacity control adds all three lambdas = 30 fits.
Total initial production work is 230 new fits. P53 omega64/lambda1 is an
additional read-only ten-case reference, not new training work.

Only full validated ten-case cohorts drive selection: highest mean Dice,
exact ties lower ASD then HD95. Exact remaining ties prefer lambda nearest1,
bandwidth nearest2 on a log scale, then smaller bandwidth/width, preventing
boundary expansion on identical scores.

After the full grid, select the best Fourier candidate and each control
independently. If the selected Fourier sigma is an endpoint, extend by factor2
in that direction and evaluate every current lambda. If any selected family
lambda is an endpoint, add the next factor10 outward and evaluate every sigma
plus both controls at that lambda. Re-evaluate only after complete cohorts.
No precision/budget changes on failure. Sigma beyond [2^-8,64] or lambda
outside [1e-10,1e10] requires review rather than uncontrolled extrapolation.

Optional followups are conditional on a conservative promising-result gate:
Fourier has strictly higher mean Dice and no worse mean HD95, ASD, foreground
std(logJ) or folding than BOTH selected controls. If the gate passes, freeze
the selected configuration and add: omega64 (10), feature counts32/128 (20),
and seeds2/3 for the Fourier winner and best plain control (40). In seed
confirmation both initialization and Fourier bank change; the coordinate
sampling stream remains seed1. This is up to 70 conditional additional fits.
If the gate fails, report the accuracy winner/Pareto trade-off and stop.

## Qualification and acceptance

Unit tests: encoding formula, immutable/nested frequency bank, parameter counts,
native initialization bounds, numerical original-coordinate derivatives and
diffusion parameter gradients, exact sampling/RNG isolation, unique names,
CLI, full-cohort selection and extension, conditional followup gate.

Four simultaneous 100-update rank0 smokes on all four GPUs: low/high Fourier
bandwidth, capacity control and plain control. Require finite gradients,
numerical FP64 derivative checks, DVF/model roundtrip and all35label metrics.
The plain100 lambda0.1 smoke must reproduce P53 initialization/sample hashes
exactly and its normalized dense field within max-absolute 1e-5; this catches
accidental changes to the baseline loss and update path.

Each new production fit must have 2500 Adam steps and 25 million network
training-coordinate evaluations, finite [160,224,192,3] fields, matching
case/direction/backend/source metadata, saved histories/checkpoints and DONE.
Canonical metrics: Dice, HD95 mm, pooled-area symmetric ASD mm, fixed-raw-positive
foreground std(logJ) and nonpositive-J percentage. Preserve all35labels/case,
presence counts and absent-label exclusions. No segmentation-driven epoch choice.

Monitor every few seconds; record process/GPU state and loss progress every15s.
Preserve failed attempts; retry CUDA OOM once. Incomplete phases cannot select
or produce acceptance. Prior controls and saved DL fields remain immutable.

## Artifacts and limitations

Accepted-protocol key: `oasis_seed3_first10_idir_fourier_sampled_diffusion_v2` beneath
`${ARTIFACT_ROOT}/runs/` and `${ARTIFACT_ROOT}/repo_experiments/`.
The v1 directory is retained as rejected qualification evidence, not production.
Keep CSV/JSON/Markdown, all label records, mean/sample-SD tables and line-only
accuracy versus foreground std(logJ) plots with VFA/SITReg references external.
Record source and reference hashes, decisions, selected configurations, runtime
and current GPU provenance. ACCEPTED and DONE require successful validation.

This is exploratory first10 test tuning, not an unbiased generalization result.
Fourier features change first-layer dimensionality and initialization scale;
capacity matching is not a proof that all optimization properties are equal.
The fixed2500 budget may under-converge some bandwidths. Smaller std(logJ) is
not evidence of fewer folds. No overall improvement claim based only on Dice.
