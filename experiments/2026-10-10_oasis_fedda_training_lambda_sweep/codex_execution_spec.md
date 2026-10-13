# Approved FeDDA-1MF training lambda sweep

## Scope and placement

Fresh original FeDDA-1MF trainings at lambda0.1,1/3,3,10, not the later101-epoch
onepass tune_1ncc checkpoint used for Feature-MSE. Historical lambda1 remains
read-only. User approved this plan on2026-10-10.

| GPU | Lambda | Dependency |
|---:|---:|---|
|0|3|Successful CPU contracts and GPU0 preflight|
|0|10|FeDDA lambda3 training AND first100 evaluation complete|
|1|0.1|P50 VFA lambda0.1 training AND first100 evaluation complete|
|2|1/3|P50 VFA lambda1/3 training AND first100 evaluation complete|

GPU3's VFA lambda3-to-lambda10 queue stays unchanged. One new worker per GPU;
never stop, reprioritize or modify existing jobs. Failed runs/logs are preserved;
no automatic retry, precision reduction or protocol changes are authorized.
Preflight needs24GiB free; production requires the larger of24GiB and observed
preflight reserved peak plus2GiB. Recheck memory at each launch and pin by UUID.

## Native training protocol

Architecture: original exp_midl_0407/fedda_1mf shared CNN encoder and pyramidal
convolutional decoder,2854079 trainable parameters,use_batch_parallel=False.
Model is unidirectional; TRAINING uses two sequential Adam updates per pair,
forward then reverse with updated weights. This is not SITReg's joint update.
Seed42,fresh random initialization,batch1,FP32,workers0,pinned memory,persistent
shuffled iterator.300 training subjects,30 validation pairs,canonical split_v1
manifest SHA2561e358ffc565c651e381a7eacb5b4e6d139b9a4a899c5819d26a58644b5c78d7a.
Preserve native raw intensities,LIA-to-RAS geometry,160x224x192 unit-mm grid.

200 epochs,50 pairs/epoch,100 directional updates/epoch,20000 Adam steps total.
Adam AMSGrad LR1e-4,betas0.9/0.999,eps1e-8,zero decay,constant LR. Full-image
NCC_vfa(window9,keep_sign=False):1-minus squared local correlation,plus lambda
times unmasked total voxel-DVF Grad3d(l2). No feature similarity,segmentation
loss,AMP,scheduler,early stopping or loss normalization. Native forward-only
30-pair validation each epoch; select best full-precision validation Dice.
Correctly recompute total validation loss; do not copy stale historical logging.
Save best/latest checkpoints,RNG states,pair identities and direction losses.

Explicit historical CUDA override: benchmark=True,deterministic=False,
cudnn TF32=True,matmul TF32=False. CPU threads8. No bitwise-training claim.

## Verification before production

Strict-load dice0.8141_epoch0196.pth.tar,reference SHA256
fa9ba93e5ea9296502f1e144accf715738fe9602cfeee32a41a81b1393a5c01e.
Reference best validation Dice0.814109588187874; optimizer step19600.
Reproduce rank0 fixed0333/moving0115 saved field: max<=0.1voxel,RMS<=1e-3,
Dice delta<=5e-5,HD95<=1e-3mm,ASD<=1e-4mm,foreground SDlogJ delta<=1e-3,
folding delta<=0.01percentage points. Full30 validation Dice delta<=1e-4.
The original VFA-derived max1e-3/RMS1e-4 field gate failed; a bounded backend
audit established fresh-process native-benchmark variation (max0.00156-0.05273,
RMS2.23e-5-3.62e-4). The accuracy/validation gates remain unchanged. See the
technical log for diagnosis and preserved failures. This is numerical,not
bitwise,reproduction; no training/backend/precision setting was changed.
Do not relax further failures without diagnosis. Test exact serialization,finite native
DVF geometry,35 per-label rows and absent-label rules. Fresh full-resolution
forward/reverse pair updates at all four lambdas,then repeated warm updates;
measure optimizer-state training peaks,finite gradients and checkpoint roundtrips.

Qualification after the bounded diagnostic: the second preflight failed the
HD95 gate despite a small field discrepancy. The subsequent non-accepting audit
completed every check with no issues and reproduced the saved field and all
five canonical metrics exactly. A separate CPU certification verifies that
saved field bitwise against the original, rechecks the validation, four lambda
smokes, checkpoint contents, memory and source hashes, and links both failed
preflights. It produces a new qualification directory; it does not alter the
audit or failed runs. This establishes adapter correctness, not cross-process
repeatability of historical benchmark mode. Accuracy tolerances were not
relaxed. The historical backend remains unchanged, and future inference is not
guaranteed to reproduce a saved field bitwise or pass every tight metric gate.

CPU tests cover loss/update equivalence,CLI,new names,dependency completion,
memory gate,source identity and detailed evaluator reuse. All workers require
the preflight initial-state hash and matching source/config/input hashes.
New FeDDA-only source files; never edit hash-locked P46/P49/P50 helpers/native
source while those workers run. Container UID:GID1009:1020.

## Evaluation and reporting

Each training produces canonical first10 fields from its best-validation
checkpoint,then an automatic separate first100 evaluation reuses those bytes
and infers ranks10-99. Canonical seed3 shuffled pair_test_200 ranks0-99.
Save per-case and all35 per-label Dice/HD95/pooled-area symmetric ASD plus both
directional ASDs,explicit absent-label exclusions,field/warped-seg outputs and
hashes. Foreground is fixed raw-image-positive; SD(logJ) uses positive J,
folding is J<=0 percent with canonical crop2. Include diffusion diagnostics.
Require100 cases and3500 label rows per candidate; aggregate completed runs only.
Produce CSV/JSON/Markdown and line-only accuracy-versus-foreground-SDlogJ curves
for five FeDDA weights with original VFA/SITReg references. Report mean/sampleSD.
Lambda1 is historical,not a fresh seed42 replication; no test-based checkpoint
selection or statistical-superiority claims. Runtime is diagnostic under load.

Run root:${ARTIFACT_ROOT}/runs/oasis_fedda_ncc9_diffusion_training_lambda_sweep_v1/.
Separate preflight sibling:oasis_fedda_ncc9_diffusion_training_preflight_v1/.
Report root:${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_fedda_training_lambda_sweep_v1/.
All generated artifacts stay outside Git. Existing references remain read-only.
