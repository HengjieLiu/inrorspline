# Pre-execution audit: six-way OASIS NCC9/diffusion experiment

Status: user approved the recommendations and implementation/execution on
2026-10-08. This document records the read-only audit before implementation;
passing the audit is not a claim that new registrations have completed.

## Settled convention

All optimized displacements, including dense parameters, use normalized-grid
XYZ units. Stage learning rates are 0.03, 0.01, 0.003, 0.001. Expand controls
before regularization and convert the expanded displacement to evaluation-grid
voxel units inside autograd. The objective is unnormalized VFA squared local
NCC9 (1 minus mean squared correlation) plus lambda times unmasked forward-valid
L2 diffusion of the total displacement. No lambda rescaling is needed.

## Issues and approved resolutions

| Issue | Evidence / consequence | Resolution in the new experiment |
|---|---|---|
| Mixed optimizer units | Prefix uses normalized controls; previous feature/raw control uses voxel parameters at LR0.3. | Normalized parameters throughout; LR0.03/0.01/0.003/0.001. |
| Adam is not exactly scale invariant | Component-specific coordinate factors and additive epsilon affect updates. CPU toy tests verified that LR-only conversion can diverge with small gradients. | Avoid optimizer reparameterization between stages; fresh Adam, default betas and epsilon1e-8, no weight decay. |
| Wrong diffusion units | Regularizing normalized displacement directly changes component weights by factors6320.25/9120.25/12432.25 at native resolution. | Convert expanded XYZ displacement using evaluation-grid (W-1,H-1,D-1)/2 before finite differences. Never use control-grid dimensions. |
| Historical pyramid geometry | Image interpolation uses align_corners=True, whereas integer domain-CPS expansion samples native indices0..188 rather than0..191 at downsample4, and0..190 at downsample2. A synthetic ramp gave0.358128/0.238750 native-voxel field differences. | Evaluate the continuous cubic basis at the exact native coordinates of the stage image grid; do not approximate with trilinear control resizing or resampled dense fields. Preserve original knot insertion. |
| Pair-minmax intensities | Old spline loader rescales inputs; native DL OASIS loader does not. Rank0 quarter-resolution identity NCC9 was0.598133 with stored intensities versus0.592610 after pair-minmax. | Raw stored float32 intensities after the same orientation transform. |
| Wrong NCC family | Old global signed NCC and feature-local signed NCC differ from squared VFA NCC9. Native VFA library also exposes an unshifted negative-CC version. | Reproduce training NCC_vfa: window9, zero-padded fixed-denominator means, FP32 epsilon clamp on variance product, loss1-mean(cc). |
| Initial-loss normalization | Previous feature/raw controls divide similarity by its initial value. | No similarity normalization and no hidden equivalent-lambda transformation. |
| Image padding | common.warp_full defaults to zero padding; audited DL sampling uses border padding. | Explicit border image sampling with align_corners=True; preserve canonical zero-padded nearest-neighbor label evaluation separately. |
| Wrong regularizer | Historical prefix uses analytic B-spline bending energy; generic native spline reports prefer analytic derivatives. | Same expanded-field finite-difference diffusion objective and diagnostic for all representations. No BE or analytic diffusion training path. |
| Frozen old prefix | Old prepare helper hard-codes lambda1000 and rank0 stopping125/150/175; old caches contain incompatible images/features/objective. | Fresh zero-start prefix per case/lambda/resolution; 140 new prefixes; no historical stopping-epoch assertion or old cache reuse. |
| Loss/state misalignment | Old prefix logs pre-update loss while retaining post-update controls. | State-indexed histories0..N, exactlyN updates, verified objective of saved state. |
| Terminal checkpoint handoff | Historical stopping regards deterioration as insufficient improvement and exports terminal controls. | Save terminal diagnostics but hand off the lowest-total-loss state, including initialization, at every stage. Never select epochs by segmentation. |
| Effective patience | Historical cps8 patience20 becomes25 due to window25. | Record requested and effective patience; preserve effective camera-ready stopping schedule. |
| New-loss stopping suitability | NCC9 includes constant loss1 on all-zero windows; old global-NCC stopping behavior need not transfer. | Record data/reg/weighted-reg and best/terminal losses. Keep approved budgets and thresholds; do not silently tune them. |
| Backend defaults | Generic runner defaults to cuDNN benchmark=True/deterministic=False. | Explicit F/T, seed1, FP32, no autocast, TF32 off; record CUDA grid_sample backward nondeterminism caveat. |
| Legacy reporting | Old evaluators emit a global-NCC diagnostic; old search has96-fit ceiling and held-out ranks1-9 wording. | Separate objective values from canonical metric diagnostics; fixed420-output matrix, all first10 exploratory. |
| Cache identity / dirty source | Rank-only cache keys and commit-with-dirty-suffix cannot identify this experiment. | Include lambda/resolution/settings/source hashes; hash actual numerical files; preserve prior artifacts. |
| Metric exclusions | Canonical metrics omit labels absent in fixed, original moving or warped moving. | Preserve convention, report excluded labels and counts, never silently omit nonfinite present-label metrics. |
| SD(log J) exclusions | Log spread excludes nonpositive Jacobians. | Report folding and positive-J fraction separately; ellipses are marginal sample SD, not confidence intervals. |
| Unequal pipeline work | 842 terminates before either final refinement. | Describe whole-pipeline comparisons, not a pure representation-capacity or equal-compute test. |

## Checks already performed before implementation

- All ten prescribed pairs and40 image/segmentation headers passed; native
  reoriented shape160x224x192 and1mm isotropic spacing.
- Normalized/voxel parameterizations gave identical synthetic objective values;
  chain-rule gradients agreed to6.1e-8 relative L2 error.
- Diffusion matched native VFA exactly in the CPU synthetic check.
- Scaling and B-spline expansion commuted within2.3e-8 voxel at cps8/4/2/1.
- Original accepted VFA/SITReg report has20 cached reference evaluations.
- GPU memory queries and artifact-space checks passed at audit time; current
  state must be recorded again at each launch. No existing jobs were changed.

## Production gate and outcome

Native-loss forward/gradient parity, aligned-basis parity and prolongation,
checkpoint/state and stopping tests, CLI/cache/legacy tests, six full-geometry
one-step GPU smokes through serialization and canonical metrics, strict finite
and metadata checks, and immutable baseline/source verification were required.
These checks passed on2026-10-08:39 discoverable OASIS tests, including16 new
tests, and all six GPU smokes. Native GPU loss/gradient parity differences and
dense initialization differences were zero; maximum knot-insertion difference
was1.662e-6 voxel in FP32. See technical_log.md for the execution record.
No scientific outcome is guaranteed by these technical checks.
