# Approved VFA training and detailed first100 evaluation

## Scope and final placement

After the same physical GPU's P49 SITReg first100 evaluation exits cleanly,
start fresh VFA training and automatically evaluate its best-validation
checkpoint on canonical seed3 shuffled ranks0–99. Final user-approved mapping:

| GPU | Lambda | Special handling |
|---:|---:|---|
|0|10|Try immediately despite tight memory; one OOM retry appended to GPU3|
|1|0.1|Wait for SITReg evaluation completion|
|2|1/3|Start after completed SITReg evaluation|
|3|3|Start after completed SITReg evaluation; lambda10 fallback runs afterward|

GPU0 override bypasses the conservative42GiB training headroom gate, but retains
an allocator cap based on current free memory minus1GiB for CUDA context. Do
not alter other jobs, precision, batch size or scientific protocol. Preserve
the failed GPU0 attempt and log. Retry lambda10 once onGPU3 after its lambda3
training AND evaluation finish; restart fresh if training failed, or retry only
evaluation if the trained checkpoint already succeeded. Other failures do not
trigger automatic retries. Other GPUs retain42GiB training/24GiB evaluation
free-memory checks. A slot with insufficient memory waits without intervention.

## Frozen native protocol

P45 audited bs3_vfa_v0,2014827 parameters, learned beta initialized1.0 and no
separate beta-evolution stage. Seed42 fresh weights, OASIS split_v1 with300
training subjects/30 validation pairs; batch1, FP32, workers0, pin_memoryTrue,
persistent shuffled training iterator, native LIA-to-RAS/raw stored intensities.
200 epochs,50 distinct random pairs/epoch. Each pair performs TWO sequential
Adam updates: moving-to-fixed, then fixed-to-moving with the updated weights.
100 updates/epoch,20000 total. Constant LR1e-4, Adam AMSGrad, betas(.9,.999),
eps1e-8, zero weight decay. NCC_vfa(window9,keep_sign=False),1 minus squared
local correlation, plus lambda times unmasked total-voxel-field Grad3d(l2).
No AMP, similarity normalization, scheduler, early stopping, masks or label loss.

Native forward validation on all30 pairs every epoch; labels1–35 nearest-warp
Dice convention retained for selection. Recompute total validation loss correctly.
Save full-precision best-validation-Dice and latest checkpoints, RNG states,
per-direction/per-pair histories, epoch losses/timings/GPU peaks and pair hashes.
Historical backend override: cuDNN benchmarkTrue/deterministicFalse,
cuDNN TF32True/matmulFalse. CPU threads8 per worker; no bitwise CUDA claim.

## Preflight and safety

Read-only lambda1 checkpoint dice0.8115_epoch0171.pth.tar SHA256
e5353f79a7e0ce0a42eb2c06f2e35bb40d18e0a538d9ae449f418182dc9138d7.
Strict loading; rank0 fixed0333/moving0115 native inference against the saved
field: max error<=1e-3voxel, RMS<=1e-4, Dice delta<=5e-5, HD95<=1e-3mm,
ASD<=1e-4mm. Full30 validation Dice within1e-4 of reference checkpoint.
Do not relax a failed gate without diagnosing/documenting its cause.

Full-geometry rank0 serialization/detail smoke: exact saved-field roundtrip,
canonical per-label output and35 label rows. Four fresh native pair updates
(eight directional Adam steps), finite losses/gradients and exact checkpoint
roundtrips. CPU tests check directional-update equivalence, final assignment,
memory override scope, OOM retry policy, prior-evaluation gating,20000-update
acceptance and exact shared label-metric functions. Freeze complete source,
input, checkpoint, pair-file and container provenance; verify source each epoch.

## Automatic evaluation

After training, canonical first10 fields are generated from the best-validation
checkpoint. A separate queued process reuses their exact bytes and evaluates
all100 cases, inferring ranks10–99. No test-selected checkpoint or training
changes. Reuse P49 label/summary functions without editing P46/P49 files.
Save all35 label rows/case with Dice, HD95, symmetric pooled-area ASD, both
directional ASDs, label voxel counts and explicit absent-label exclusions.
JSON null/CSV blank for absent labels; unexpected numerical failures stop.
Save DVFs, warped segmentations, per-case/label CSV+JSON, aggregate mean/sampleSD,
foreground SD(logJ), folding%, diffusion and full-volume Jacobian diagnostics.
Four complete evaluations yield400 cases and14000 label rows.

UID:GID1009:1020 in the audited artifact-store container. Artifacts:
${ARTIFACT_ROOT}/runs/oasis_vfa_ncc9_diffusion_training_lambda_sweep_v1/;
preflight in separate sibling oasis_vfa_ncc9_diffusion_training_preflight_v1/.
First100 evaluations live under the training queue root's first100/ subdirectory.
Existing jobs, source files, reference fields and prior outputs stay untouched.
No new lambda1 training is authorized. Exploratory test comparisons and shared
GPU timings are not unbiased model-selection or controlled runtime claims.
