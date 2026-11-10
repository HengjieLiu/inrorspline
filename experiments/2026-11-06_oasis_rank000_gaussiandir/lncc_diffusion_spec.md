# P83 follow-up: unsupervised LNCC9 squared + diffusion, lambda 1

Status: **deferred by the user's paper-loss-first decision**. Prototype numerical
tests and both native smokes passed; no full standardized-loss runs launched.
The active follow-up is [paper-loss convergence and resources](convergence_spec.md).
The specification below records the preserved prototype, originally requested2026-11-06.

The user requested image-only registration with the previously studied
`1-LNCC² + lambda*diffusion`, lambda1. Run both existing GaussianDIR variants
independently from identity on rank0 fixed0333/moving0115. The earlier supervised
fits and signed-global-NCC/TV registrations remain immutable reference rows.
Configuration: [protocol_lncc9_diffusion_v1.json](protocol_lncc9_diffusion_v1.json).

## Known limitations

- One repeatedly explored pair, one seed; exploratory comparison only.
- This changes the GaussianDIR objective and training sampling domain. It is
  explicitly a loss adaptation, not the released GaussianDIR training recipe.
- Gaussian populations, thresholds, Adam group rates and adaptation ordering
  retain the preceding variant definitions. Thresholds are not recalibrated to
  this loss. Actual population history and invalid-population failures are recorded.
- Both methods retain Euclidean KNN support and direct displacement fields;
  neither guarantees continuity across neighbor changes or positive Jacobians.
- The stochastic loss estimates the full-volume objective. It does not compute
  a dense full-volume gradient at each step. The exact final dense objective is
  evaluated independently; stochastic convergence is not a certified optimum.

## Frozen objective and sampling

Reuse P43's definitions in
`scripts/experiments/oasis_alpha_sweep/dl_ncc_diffusion.py`:

```text
L = 1 - mean_p[cov_9(F,W)^2 / max(var_9(F)*var_9(W), FP32_epsilon)]
    + 1 * (1/3) * sum_axis mean_valid_edges,components[(delta_axis u_voxel)^2]
```

Native160×224×192 images retain stored intensities and the verified orientation.
Each9×9×9 window uses population moments, zero padding and divisor729, including
boundary zeros. Epsilon is1.1920928955078125e-7 on the variance product, with no
additive epsilon, initial-loss normalization or extra similarity multiplier.
The moving image is warped trilinearly with border padding, matching P43.
Normalized GaussianDIR displacement is converted explicitly to canonical
align-corners-True sampling coordinates and to voxel units for diffusion.
The diffusion acts on the **total coarse-plus-fine field**, including the frozen
coarse field, not just the active residual or Gaussian parameters.

At each step,128 window centers are drawn from a uniformly shuffled full-box
voxel permutation. Evaluate every neighbor's own GaussianDIR displacement
in each complete window. Outside-image window entries are zero even if a
displaced moving-image coordinate would be valid. Independently draw5,000 valid
forward edges per axis, uniformly with replacement on that axis's complete
edge domain. Evaluate both endpoints and average squared voxel differences over
components, sampled edges and axes. Both terms are unbiased estimates of the
corresponding full-volume reductions, up to FP32 arithmetic. Nominal Gaussian
queries per step:128*729 +3*2*5000 =123,312.

There is no foreground mask in either loss term. Node initialization/caps still
use raw fixed `image>0`, retaining the previous representation's initial geometry.
Segmentations and SITReg fields are only loaded for immutable-input verification
and posthoc evaluation; neither enters the training objective or selection.

## Other settings and comparison

Retain seed6,K20, variant LRs,50-step warmup/cosine, adaptive thresholds. Control
uses2,000 single-scale steps; corrected reconstruction uses1,000 coarse then
1,000 fine steps with frozen coarse and zero-motion additive fine initialization.
No image pyramid or composition. The1,200-second active watchdog remains; any
timeout is labeled and stage completion is reported. Final step selection uses
neither label metrics nor noisy lowest-batch loss.

Use the same complete full-box accuracy/regularity evaluator as P83:35-label
Dice/HD95/ASD with eligibility, image metrics, full/foreground Jacobian, diffusion,
BE and NDV, plus field/Jacobian agreement with SITReg as reference diagnostics.
The common evaluator retains its zero-padding convention for historical metric
comparability; the separately recorded LNCC training objective uses border
padding. Compare new LNCC/diffusion rows with identity, SITReg and both old TV rows.

## Verification and artifact plan

Before production, verify complete-window NCC values/gradients against unmodified
P43 NCC9, complete loss/displacement gradients against P43 StageObjective, edge
domain coverage, boundary divisor/epsilon/sign behavior, and sampler/RNG replay.
Run the existing13 representation tests, then native-volume smokes for both new
variants with real adaptation and both corrected stages. Recheck occupancy and
image/runtime identity; retain cuDNN benchmarkFalse/deterministicTrue and TF32False.

Generated runs, immutable source snapshots and checkpoints:
`${ARTIFACT_ROOT}/runs/oasis_rank000_gaussiandir_lncc9_diffusion_v1/`.
Generated tables, figures, report and executed notebook:
`${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_lncc9_diffusion_v1/`.
Use `smoke_v1` then `pilot_20261106_v1` execution IDs and fresh output directories.
The base environment, upstream submodule, historical P83 config and outputs are
unchanged. No further parameter sweep is included in this follow-up.
