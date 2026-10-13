# P44: Full-resolution native DL-loss sweep

Approved execution: 2026-10-08. Variant:
`oasis_seed3_first10_mrdbscp842_dense_fullres_dl_loss_sweep_v1`.

350 independent registrations: five similarities × seven diffusion weights ×
seed3 shuffled OASIS ranks0–9. Rank0 fixed0333, moving0115; moving-to-fixed pull
sampling. All ten cases are exploratory, not an unbiased test benchmark.

## Objective and weights

Use `s * native_similarity + s * effective_lambda * diffusion` at every stage.
Multipliers: `1/100, 1/10, 1/3, 1, 3, 10, 100` (no rounding of thirds).

| Similarity | DL-selected effective center | Similarity scale s | Native definition |
|---|---:|---:|---|
| MSE | 0.005 | 1 | Full-image mean squared error |
| MIND | 0.04 | 25 | MINDSSC radius2/dilation2, 12 channels, descriptor MSE |
| RC | 0.1 | 10 | Residual Complexity: orthonormal DCT-II, mean log1p(coeff²/0.05) |
| NGF | 0.2 | 1 | Native 3D 1-minus-squared gradient correlation, epsilon1e-5 |
| Feature-MSE | 0.2 | 1 | Frozen FEDD-A first-level encoder MSE after image warping |

Native implementation imported directly from wavereg. No mask or initial-loss
normalization. Fixed descriptors may be cached after native value/gradient parity.
Moving MIND/Feature-MSE descriptors are computed from the warped image, not warped
cached descriptors. Feature-MSE uses checkpoint `dice0.8112_epoch0093.pth.tar`,
encoder projection+block0; this is not the earlier VFA projected-feature loss.
Frozen weights still propagate input/field gradients.

## Geometry and optimization

Fresh zero cps8 start for each registration; no NCC-prefix reuse. Exact dyadic
cps8→4→2 prolongation then one expansion and exact detach/clone into a dense field.
No cps1 controls, analytic bending energy, DVF fitting or segmentation supervision.
Full `[160,224,192]` image support at every stage, raw stored intensities, existing
LIA→RAS orientation, border sampling, align_corners=True. Parameters normalized xyz
throughout; convert total field to voxel xyz within autograd for unmasked forward-
valid diffusion (squared differences, component mean, axes mean).

| Stage | LR | Max/min epochs | Requested/effective patience |
|---|---:|---:|---:|
| cps8 | .03 | 400/40 | 20/25 |
| cps4 | .01 | 800/80 | 35/35 |
| cps2 | .003 | 1300/130 | 50/50 |
| dense | .001 | 2000/300 | 100/100 |

Reuse P43 Adam/median-window logic unchanged: fresh Adam per stage, betas.9/.999,
epsilon1e-8, no weight decay; checks25/window25; relative total/data1e-4/5e-5.
No prefix LR reductions; dense two halvings at least100 updates apart then stop
on next plateau. Best-total-loss state at every stage includes initialization.
State-indexed history, best/last checkpoints, last Adam state and complete provenance.
Seed1 FP32, cuDNN benchmarkFalse/deterministicTrue, TF32off. CUDA grid-sample
backward retains a nondeterminism limitation.

## Execution and acceptance

Existing artifact-store container, UID:GID1009:1020; GPUs0–3, max4 added workers,
one per GPU; existing workloads untouched. Descriptor jobs first. Fresh GPU state
logged at each launch; memory gate uses measured smoke peaks plus4GiB. Preserve all
failures; retry CUDA OOM once after a released worker, on the most-free suitable GPU.

Unit contracts plus five native-parity/full-geometry one-update-per-stage smokes
gate production. Verify canonical metrics, exact dense initialization, finite DVFs,
case/direction, full-volume objective counts, frozen encoder and loss/gradient parity.
Native MIND constant-zero behavior is diagnosed but not silently modified.

Reports require350 successful outputs; no incomplete aggregation. Compare against
70 immutable P43 full-resolution dense NCC outputs and20 original VFA/SITReg cases.
CSV/JSON/Markdown, means±sampleSD, per-case paired deltas, best epochs/runtime/GPU,
Dice/HD95/ASD, foreground positive-J SD(log J), foreground J≤0%, diffusion.
ARC and folding companion figures with marginal±1sampleSD ellipses, not confidence
regions. Dice-first per-loss selection (ties ASD then HD95), metric winners and
accuracy/folding Pareto candidates. Flag boundary winners, do not expand the sweep.

Run artifacts: `${ARTIFACT_ROOT}/runs/<variant>/`.
Review artifacts: `${ARTIFACT_ROOT}/repo_experiments/<variant>/`.

## DL anchor provenance and limitations

Best validation checkpoints, not last-epoch weights, informed the centers:
MSE FEDD-A w.005 Dice.8085; NGF FEDD-A w.2 Dice.7949; Feature-MSE FEDD-A w.2
Dice.8135; RC FEDD-B3 similarity10/regularizer1 Dice.8246; MIND FEDD-B
similarity25/regularizer1 Dice.8141. Different architectures/budgets, not a uniform
comparison of DL losses. Original MIND train_all_v2_2506.py is unavailable; surviving
training sources use the native MIND class and L2 Grad3d.

Native NGF current3D form is 1-correlation, not historical -.5*correlation.
RC fast-class default uses sum, so use native matrix class with explicit mean.
Preserve native similarity scaling because Adam epsilon precludes exact global
loss-scale invariance. Stop on nonfinite native objectives; do not silently mask or
change stabilizers. All input, checkpoint, configuration, numerical source and
reference hashes are recorded and checked. Previous runners/results are immutable.
