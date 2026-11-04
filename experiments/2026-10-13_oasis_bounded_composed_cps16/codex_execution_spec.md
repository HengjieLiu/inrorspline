# Rank0 cps16 bounded-composition extension (P65)

Approved single fresh registration from identity: rank0 of the repo seed3
shuffled OASIS split, fixed0333/moving0115, moving-to-fixed via pull mapping.
One bounded cubic increment per cps16/8/4/2/1, diffusion weights[1,1,1,5,1].
Preserve P64 sources, accepted outputs and all previous reference fields.

| CPS | Loss downsample | Lambda | Normalized latent LR | Max/min epochs | Requested patience |
|---:|---:|---:|---:|---:|---:|
| 16 | 8 | 1 | 0.03 | 400/40 | 20 |
| 8 | 4 | 1 | 0.03 | 400/40 | 20 |
| 4 | 2 | 1 | 0.01 | 800/80 | 35 |
| 2 | 1 | 5 | 0.003 | 1300/130 | 50 |
| 1 | 1 | 1 | 0.001 | 2000/300 | 100 |

Native endpoint-aligned cubic samples define each border-extended trilinear
elementary map. Retain all five maps; compose Phi_previous o delta_new.
No old prefix, knot insertion, accumulated-map resampling or dense refinement.
Freeze earlier samples while retaining gradients through sampling coordinates.
Controls=a*tanh(p/a), a=0.99*cps*b/native_voxel_scale_xyz; zero fresh latent,
unit derivative at zero. Derive cps16 bound in FP64 over4096 sampling phases,
eight cell corners and eight derivative sign patterns. Verify SITReg's own
half-voxel kernel lookup/native algorithm independently; do not copy its cap.

Retain raw intensity NCC9=1-mean(local correlation squared), unmasked forward
valid total-field L2 diffusion in current evaluation-grid voxel units. New
cps16 loss grid20x28x24 uses the historical Gaussian antialias sigma4/radius12.
No analytic regularizer, segmentation objective or reference supervision.
Fresh Adam per stage; seed1, FP32, cuDNN benchmarkFalse/deterministicTrue,
TF32off. Check/median25, effective patience>=25, thresholds1e-4/5e-5.
Only cps1 (now index4) gets two LR halvings with100 epochs between actions.
Each stage retains lowest-total-loss state including initialization.

Separate runner/module/method identity, explicit five-stage manifest and
unique names. Sources, inputs, split, checkpoints, fields and reference
reports are hashed. Tests cover old-helper parity, schedule/indexing, native
geometry, bound witnesses, tanh derivative, loss/gradient and composition.
Full-size one-step-five-stage smoke gates the one production registration.
Require checkpoint/field/loss reconstruction, canonical absent-label handling,
finite160x224x192x3 output, all elementary corner derivative row sums<1,
same100000 seed1 foreground off-grid probes as P64 for chain versus export.

GPU3 only, smoke then production sequentially, UID:GID1009:1020. Recheck
memory at each launch; require8GiB or smoke peak+2GiB, whichever is larger.
Existing jobs untouched. Preserve failures and stop for audit; no auto retry,
extra increments, lambda/LR sweep or first-ten expansion.

External run/report key under ARTIFACT_ROOT:
oasis_seed3_rank000_mrdbscp168421_tanh_composed_lambda11151_v1.
Compare five accuracy/regularity metrics with P64 bounded8421, P63 unbounded,
P62 dense, original VFA/SITReg. Save CSV/JSON/Markdown, paired deltas, per-stage
metrics, saturation>95%, stopping/best epochs, exact bounds and runtimes.
Single exploratory case: no across-case SD, causal proof or export guarantee.
