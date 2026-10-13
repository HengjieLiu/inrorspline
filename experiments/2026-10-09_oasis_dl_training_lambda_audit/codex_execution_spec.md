# OASIS VFA and SITReg training lambda audit

This preserves the original audit-only scope. Subsequent user approval for
four-GPU SITReg training is recorded separately in P46; subsequent VFA training
and detailed first100 evaluation approval is recorded in P50.

## Authorization

Audit historical lambda1 training and plan fresh lambda0.1, 1/3, 3 and10 for
both models. Full training is NOT authorized. Only short sequential training
memory smokes on GPU3 are authorized; existing jobs and data remain unchanged.

## Historical protocol and proposed sweep

- Model keys:`bs3_vfa_v0`, `bs4_sitreg_v0`; fresh initialization, seed42.
- OASIS split_v1:300 training subjects,30 validation pairs,200 test pairs;
  subject-disjoint. Dataset manifest SHA256:
  `1e358ffc565c651e381a7eacb5b4e6d139b9a4a899c5819d26a58644b5c78d7a`.
- Native loader pairs distinct random training subjects, shuffles moving
  subjects and keeps its iterator across nominal epochs. Batch1, workers0,
  pinned memory. Native LIA-to-RAS transpose/flips, shape160x224x192,
  raw stored intensities, no additional augmentation or normalization.
- Adam, constant LR1e-4, betas(.9,.999), eps1e-8, weight_decay0, amsgrad=True.
  FP32, no AMP, no scheduler.200 epochs,100 directional samples/epoch.
- VFA:50 pairs/epoch and100 separate directional Adam updates.
  SITReg:50 pairs/epoch and50 joint bidirectional Adam updates.
- Native `NCC_vfa(window_size=9)` is1 minus mean local SQUARED correlation,
  not signed global NCC. Full image, no similarity normalization. Native
  `Grad3d(penalty='l2')` on total voxel-unit displacement, unmasked valid
  forward differences, averaged over components and axes. Loss=NCC9+lambda*R.
  SITReg averages both directions for each term before one backward pass.
- Validation30 pairs every epoch; choose best validation Dice, not test Dice.
  Historical `epoch_early_check=100` separates checkpoint directories, NOT
  early stopping. Keep best-Dice and latest checkpoints in the future sweep.
  Preserve the native checkpoint-selection Dice over labels1..35, nearest
  segmentation warping and its epsilon/absent-label convention; separately
  compute canonical posthoc test metrics for comparison with existing reports.
- Historical backend override:cuDNN benchmark=True, deterministic=False.
  Seed42 is not a bitwise reproducibility guarantee. Record effective TF32.
  This differs from the deterministic/TF32-off per-pair optimization sweeps;
  do not interpret cross-pipeline runtime as an algorithm-only comparison.
- Reference checkpoints:`dice0.8115_epoch0171.pth.tar` for VFA and
  `dice0.8145_epoch0191.pth.tar` for SITReg. Filename Dice is validation Dice.
- Proposed eight fresh runs change only model/lambda, NOT fine-tuning of
  lambda1 weights. Use exact float1/3 with a collision-free1over3 run label.
  Match initial-weight hashes and training pair sequences across lambdas
  within each model. Totals:VFA20000 Adam updates; SITReg10000 updates.
- After approval, add a scoped explicit-path training adapter, preserving
  native update/data semantics and fixing validation-total logging. Gate
  production on saved-checkpoint inference parity and native loss/update tests.
  Any new lambda1 bridge training requires approval if source drift demands it.
- Proposed evaluation:first10 seed3 shuffled pairs; canonical Dice/HD95/ASD,
  fixed-image-positive SD(logJ) and folding, means/sampleSD, line-only ARC
  curves beside prior per-case optimization sweeps. All first10 remain
  exploratory. Full200 evaluation can be added after scope approval.

## Authorized smoke

Native wavereg factories, dataset, NCC and TransMorph diffusion. Strictly
load the reference checkpoint then restore fresh seed42 weights. Three
training pairs from fresh initialization, then three pairs with trained
lambda1 weights; fresh Adam for each phase. Discard updates and never save
trained weights. VFA exits before SITReg starts, on physical GPU3/cuda:0 only.
Batch1, full160x224x192, FP32, lambda1, LR1e-4 and historical backend override.
Keep native validation-grid buffers resident. Measure forward/backward/Adam
allocated/reserved peaks and poll GPU use; log finite gradients and timings.

Run as UID:GID1009:1020 in the default artifact-store container, with external
outputs under`${ARTIFACT_ROOT}/runs/oasis_vfa_sitreg_training_lambda_audit_v1/`.
Pass source/data/checkpoint paths explicitly. No shared dependency changes,
W&B/network logging or production launcher. Preserve failures; do not alter
batch size, precision, loss or architecture automatically on OOM.

## Audit issues and acceptance gates

1. Original April2025 configs and optimizer states survive, but W&B metadata
   contains no Git commit. Current source is not an immutable historical copy.
2. Public trainer uses`--ncc-type vfa`; current wavereg uses
   `--Lsim-type ncc-vfa` and similarity weight1. Do not substitute fast NCC.
3. SITReg upstream JSON supplies architecture only; its30 epochs, LR4e-4,
   NCC11, batch4 and LUMIR normalization are NOT the OASIS baseline protocol.
4. Public trainer has missing config/model files and stale raw-data paths;
   do not execute unchanged. Use scoped path overrides, not source edits.
5. Historical/public VFA validation omits recomputing total loss; current
   wavereg fixes it. Dice-based checkpoint selection is unaffected. Never
   select new checkpoints by the stale total-loss log.
6. Preserve persistent iterator and50-pair epochs. VFA wrapper prints
   batch_parallel=True, but the unified VFA branch encodes images separately;
   that wrapper flag does not control this branch.
7. Require strict checkpoint loading, expected parameter counts, finite
   gradients and allocated optimizer states. Record source/config hashes,
   versions, GPU state, case IDs, complete settings and checkpoint immutability.
8. Short smokes do not bound all future lambda/inversion workloads. Full
   production remains approval-gated after presenting measured results.
9. Historical checkpoint retention compares running-best metadata with
   rounded current-Dice filenames. Save explicit full-precision best and
   latest pointers in new output directories; do not reuse deletion logic
   against historical checkpoints.

## Source map

Original command notebooks and local W&B configs are under
`${WAVEREG_ROOT}/scripts/250218_oasis_v1/notebooks_wipmidl_1/`.
The historical training program was
`${WAVEREG_ROOT}/scripts/250218_oasis_v1/mycode/train_all_v1.py`;
the current file has additional loss options and is not a historical snapshot.
The public counterpart is`${DL_REVISITED_ROOT}/scripts/train_all_v1.py`.

Native modules under`${WAVEREG_ROOT}/code/`:
`utils_models_v2v2_baseline.py`, `networks/model_vfa_v0.py`,
`networks_v2/flow_estimator.py`, `utils_datasets_my.py`,
`utils_losses_ncc.py`, `utils_warp.py`, `utils_eval.py`.
Diffusion is in
`TransMorph_Transformer_for_Medical_Image_Registration/OASIS/TransMorph/losses.py`.

VFA has2014827 parameters, four encoder downsamplings,8 initial channels,
64 maximum channels,8 matching channels and learned beta initialized1.0,
without the original paper's separate beta-evolution training schedule.
SITReg has15076110 parameters; its very-deep-heavy architecture uses feature
widths12/32/64/128/128/128, GroupNorm4 and ReLU, no affine prediction,
full-resolution transformation output, control multiplier0.99, forward and
backward Anderson solvers with2..50 iterations and threshold1e-2.

Historical artifact branches under`${WAVEREG_ARTIFACT_ROOT}/wipmidl_0407/OASIS_v1/round1_MIDL/`:

- `gpu8_OASIS_v1_bs3_vfa_v0_run1_20250407145115`
- `gpu8_OASIS_v1_bs4_sitreg_v0_run1_20250407165940`

Current source revisions:inrorspline`880c9898184f95057bfbb3968ade03d121031d93`,
wavereg`546d6c760aa0cb456a46fad2f1627f6ef069531f`,
DL-Revisited`8f2128a9b9f2e3b8b39c5bd280c9c3420576b4a4`.
Dirty source hashes, not these commits alone, define the measured smoke.
Container image ID:
`sha256:dab700c415830dcf07c16ef7f2901deed95bb104eda388514c3ce74a43354ba8`.
Historical and current PyTorch both2.6.0+cu118; current CUDA runtime11.8,
cuDNN90100. Historical complete environment parity is not established.
