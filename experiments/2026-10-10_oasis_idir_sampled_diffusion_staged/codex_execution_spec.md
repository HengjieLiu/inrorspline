# Sampled IDIR: diffusion first at omega32, then frequency at frozen lambda

User-approved2026-10-10: retain old10,000-coordinate IDIR updates; sampled
signed global1-NCC plus analytic diffusion. Postpone the2Domega×lambda grid.
Use two added workers onGPU1 and two onGPU3 (slots1,1,3,3), UID:GID1009:1020.
Do not modify existing jobs, old BE runs or cancelled full-volume artifacts.

## Fixed protocol

- Canonical seed3shuffled first10, fixed/moving as the pair file specifies.
- SIREN[3,256,256,256,3], native omega-dependent initialization, seed1. Preserve
  pair-minmax images, normalized XYZ input/output, zero-padded trilinear sampling,
  align_cornersTrue and fixed raw>0 training domain. No Fourier/spline prefix.
- Every Adam update samples exactly10,000 voxel centers without replacement via
  the original torch.randperm operation; refresh every update. Log/checksum the
  complete sample sequence and enforce exactly25million SIREN training points.
- L=1+legacy_negative_global_NCC+lambda*R. R is the mean of squared analytic
  displacement derivatives over samples,3components and3axes. Convert normalized
  derivatives to voxel derivatives by s_component/s_axis, where
  s=[(W-1)/2,(H-1)/2,(D-1)/2]. create_graphTrue enables parameter backpropagation.
  No extra division by10,000, identity Jacobian, BE or full-volume/local NCC.
- Fresh Adam LR1e-4, betas.9/.999, epsilon1e-8, weight_decay0, AMSGradFalse,
  exactly2500updates, no scheduler/early stop. Primary field is terminal2500.
- FP32, cuDNN benchmarkFalse/deterministicTrue, both TF32False, explicit current
  repo default. Old checkpoint arguments did not record effective CUDA flags;
  do not claim bitwise historical backend reproduction. grid_sample backward
  can remain nondeterministic. Existing jobs are untouched.

## Calibration and adaptive stages

1. Read all60 old BE first10 summaries/checkpoints as immutable references.
   Determine the best old BE alpha by mean Dice, then ASD/HD95. At ranks0/4/9,
   evaluate fresh and best-BE SIRENs on the same10,000 coordinates:1-NCC,
   original BE (including its legacy extra/10000), analytic voxel diffusion,
   and parameter-gradient norms. These are diagnostics, not new training runs
   or an assumption that BE/diffusion lambda scales are interchangeable.
2. Qualify four concurrent100-update rank0 smokes atomega32 andlambda
   1e-3/1e-1/1e1/1e3, one per slot. Enforce point budgets, finite gradients,
   terminal checkpoint/DVF roundtrip and all35label metrics. Record timing
   under actual two-workers-per-GPU contention and peak allocated/reserved memory.
3. Atomega32, run all70fits forlambda10^e,e=-3,-2,-1,0,1,2,3 and ranks0–9.
   Only complete10case cohorts with all validation checks are selectable.
4. Select highest mean Dice, exact ties lower mean ASD then HD95. Remaining
   exact ties prefer the exponent nearest0, preventing infinite expansion on
   identical numerical plateaus. Regularity is reported, not silently used as
   an additional selection constraint. If the selected lambda is at either
   tested boundary, append the next outer decade for all10cases and repeat.
   No test-data-based early stopping or incomplete-cohort selection. If the
   numerical FP32 representable range would be exceeded, stop for human review.
5. Once the selected lambda is bracketed, save an immutable selection record;
   hold it fixed and sweepomega8/16/32/64/128 across first10. Reuse only the
   just-validated selectedomega32 cohort, adding40fresh fits. Do not retune
   lambda at eachomega. This is a sequential search, not the postponed2Dgrid.

The controller polls workers, loss progress, source integrity and current GPU
memory every few seconds; writes live state every15seconds; records every
cohort decision and boundary extension with metrics and provenance. Nonfinite
or failed runs cannot advance selection. Preserve failures, retry OOM once
without changing precision/sample budget/loss, and stop for unresolved failures.

## Metrics and artifacts

After training only, evaluate a full dense field for canonical Dice, HD95mm,
pooled-area symmetric ASDmm, fixed-raw-positive foreground std(logJ) and folding
percentage using canonical crop2Jacobian. Keep all35labels/case with presence
counts and exclusions, both directional ASDs, warped segmentation, final DVF,
weights/Adam state, histories, sample hashes and complete source/input metadata.

Save incremental CSV/JSON, complete-cohort mean/sampleSD, final Markdown and
line-only accuracy-versus-stdlogJ plots with original VFA/SITReg references.
No-extension budget:70+40=110fits,3850label rows. Each lambda extension adds
10fits/350label rows. All first10cases are exploratory test-case tuning.

Distinct artifact key:
`oasis_seed3_first10_idir_sampled_global_ncc_analytic_diffusion_staged_v1`
under`${ARTIFACT_ROOT}/runs/` and`${ARTIFACT_ROOT}/repo_experiments/`.
Cancelled full-volume sources/results are not scientific evidence for this
method. Shared utility functions may be reused read-only with source hashes.

Known limitations: this is not the DL localNCC9/full-volume forward-FD loss;
analytic diffusion can differ from sampled-grid FD, especially at highomega.
Equal2500updates need not converge equally. The chosen lambda is conditional
onomega32 and accuracy-first selection; no optimality at otheromega is implied.
GPU utilization/timing is shared-workload diagnostic, not a speed claim.
