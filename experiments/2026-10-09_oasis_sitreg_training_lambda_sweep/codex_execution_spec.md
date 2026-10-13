# OASIS SITReg training lambda sweep

## Approved execution

User approved starting SITReg on GPUs0,1,2,3 concurrently. Four independent
fresh runs: lambda0.1,1/3,3,10, respectively. VFA and additional lambda1
training remain out of scope. Existing jobs and historical outputs are read-only.

Use the P45 audited native model, preprocessing, data and training strategy:
OASIS300 training images,30 validation pairs; full160x224x192, stored
intensities, native LIA-to-RAS orientation; no train masks or segmentations.
Seed42; native initialization; batch1; Adam AMSGrad at constant LR1e-4;
betas(.9,.999), eps1e-8, zero weight decay.200 epochs,50 random distinct
training pairs per epoch, one joint bidirectional Adam update per pair.
Keep the native persistent shuffled loader iterator across epochs.
NCC_vfa(window9,keep_sign=False) plus lambda times native Grad3d(l2),
each averaged over both directions. No AMP, scheduler or early stopping.
Backend override:cuDNN benchmarkTrue/deterministicFalse, cuDNN TF32True,
matmul TF32False, matching P45's historical-mode smoke.

Validate all30 pairs after every epoch. Preserve native labels1..35 Dice
selection; save full-precision best-Dice and latest checkpoints atomically
under new directories, without historical checkpoint deletion logic.
Log every pair, epoch, losses, validation metrics and GPU peaks; save RNG
states and initialization/pair-stream hashes. Best selection never uses test
metrics. On completion, export and canonically evaluate first10 fields with
Dice/HD95/symmetric ASD and fixed-positive SD(logJ), folding and diffusion.
First10 remains exploratory. TRAINING_COMPLETED and COMPLETED are distinct.

## Gates and memory

Before launch:CPU loss/update contracts, exact rank0 IDs/direction, strict
lambda1 checkpoint loading, fresh inference versus saved field with numerical
and canonical metric agreement, native full-volume Adam and
checkpoint serialization smoke. Fail closed on a material discrepancy.
Preflight gates and source/data/checkpoint hashes are required by every worker.

The initial1e-4 maximum-voxel gate failed and is preserved in attempt1.
Unchanged-weight repeated inference in historical benchmark mode differed by
up to0.0055voxel; backend choices also changed solver trajectories. PyTorch,
cuDNN, composable-mapping and inversion-layer versions match the historical
environment, and the current default source path is equivalent to the archived
February2025 SITReg implementation. Use numerical, NOT bitwise reproduction:
field RMS<=0.005voxel and max<=0.25voxel; rank0 differences in Dice<=5e-5,
HD95<=0.001mm and ASD<=0.0001mm. Also require full30-pair native validation
Dice within1e-4 of the checkpoint's historical value. The training objective,
optimizer, model and backend remain unchanged. This documented gate revision
does not declare the legacy field exactly reproduced.

Current free memory is about38.6/37.4/43.3/47.4GiB. P45 measured36.48GiB total
SITReg startup use. The explicit four-GPU request uses a tighter budget than
P45's conservative39GiB recommendation:require at least37GiB free and cap each
worker's PyTorch allocator at initial free memory minus1GiB, allowing for CUDA
context/workspace overhead. GPU1 has less than1GiB headroom above observed
total startup; later workload growth can still cause OOM. Do not change
precision, architecture, batch size or losses to conceal an OOM. Preserve
failed runs and let unaffected runs continue; no automatic OOM restart.
Never stop, reprioritize or alter other GPU jobs.

Container UID:GID1009:1020, default artifact-store image. One worker per GPU,
pinned by UUID. Separate processes are launched together, not a shared-model
distributed training job. Source hashes and startup memory state are recorded.
Generated outputs only under
`${ARTIFACT_ROOT}/runs/oasis_sitreg_ncc9_diffusion_training_lambda_sweep_v1/`.
No W&B uploads, shared dependency installs or native source edits.

## Known limitations

No immutable April2025 source commit exists; checkpoint inference parity and
current source hashes constrain source drift. Non-deterministic CUDA kernels
mean equal seeds are not bitwise reproduction. Shared GPU workloads affect
runtime. Lambda1 reference retains its historical training trajectory.
