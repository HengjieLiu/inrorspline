# Approved detailed first100 evaluation follow-up

User approved evaluating all four new SITReg checkpoints on the same seed3
shuffled first100 pairs, starting the completed lambda10 immediately and
queuing lambda0.1/1over3/3 until their training finishes. User explicitly
requires per-segmentation per-case Dice, HD95 and symmetric ASD, not only means.

## Immutable inputs and scheduling

Do not edit P46's running training sources, checkpoints, first10 fields or
existing jobs. New standalone evaluator and queue use each run's final
best-validation-Dice checkpoint, never test-selected weights. Require
TRAINING_COMPLETED, completed first10 evaluation, final200-epoch/10000-update
summary and clean training process exit before scheduling on the same GPU.
One evaluation worker per released training slot, preserving other workloads.
Fresh free-memory checks and allocator caps apply; wait for memory if needed.
No retraining, VFA inference or lambda1 evaluation is included in this approval.

Run as container UID:GID1009:1020. Preserve the explicit historical backend
override: cuDNN benchmarkTrue/deterministicFalse, cuDNN TF32True/matmulFalse,
FP32, native stored intensities and LIA-to-RAS orientation. Frozen model,
no gradients. Numerical rather than bitwise reproducibility remains a limitation.

## Evaluation and artifacts

Pair file: configs/splits/oasis/pair_test_200_shuffled_seed000003.txt, ranks0–99,
first subject fixed, second moving. Preserve existing first10 fields byte for
byte; re-evaluate their complete label details and assert case-mean parity.
Infer ranks10–99 from the same best-validation checkpoint and save finite
channel-first[3,160,224,192] voxel-zyx NIfTI fields with exact serialization
roundtrips. Save warped segmentations for inspection, without using labels
for model inference or checkpoint selection.

Reuse the unchanged common.py canonical accuracy and regularity functions:
labels1–35; nearest label interpolation;1mm spacing; area-weighted surface
HD95 and pooled-area symmetric ASD, plus both directional ASD values.
A label must exist in fixed, moving and warped segmentation to enter the case
mean. Save all35 rows regardless; excluded values are JSON null/CSV blank,
with presence counts and explicit exclusion reason. Unexpected nonfinite
metrics for present labels fail evaluation. Assert label means reconstruct
the canonical case metrics exactly.

Save per-case wide CSV/JSON, long per-label-per-case CSV/JSON (3500 rows/run),
per-label cohort summaries, case-weighted mean/sample SD summaries and Markdown.
Foreground SD(logJ), nonpositive-Jacobian percentage and total-field diffusion
use the unchanged canonical fixed-raw-positive pipeline. Preserve full-volume
Jacobian diagnostics as well. Each case includes IDs, rank, lambda, checkpoint
hash, input hashes, field hash/path, GPU, runtime and a completion marker.
Only complete100-case cohorts receive final summaries and completion markers.
Queue aggregates all four only after400 successful case evaluations.

Artifacts: ${ARTIFACT_ROOT}/runs/oasis_seed3_first100_sitreg_training_lambda_eval_v1/.
Smoke: separate sibling oasis_seed3_first100_sitreg_training_lambda_eval_smoke_v1.
Failed smokes are preserved; the accepted follow-up uses the sibling smoke_v3 key.
Partial results/failures remain preserved; no automatic retry or overwrite.

## Preflight

CPU contracts cover CLI, completion gating, exact canonical pair order, all35
label rows, absent-label semantics, JSON/CSV serialization, numerical failure
handling, label-to-case reconstruction and sample-SD aggregation.
Full-geometry lambda10 GPU3 smoke evaluates reused rank0 and freshly inferred
rank10 through DVF serialization, labels, accuracy and regularity. Production
queue requires its success and unchanged evaluator/native source hashes.
Production memory threshold is max(24GiB, smoke reserved peak+3GiB).
Reused-field metric tolerances are1e-10 absolute for accuracy/Jacobian metrics
and1e-8 for float32 diffusion reductions, whose summation order can change
with NIfTI array strides. The first smoke reproduced all accuracy/Jacobian
metrics exactly but failed the original uniform1e-10 check on a4.66e-10
diffusion difference; only the diffusion numerical tolerance was corrected.

## Interpretation

This extends a previously explored test cohort; do not claim unbiased model
selection from these test results. Runtime remains diagnostic with unrelated
jobs active. Label-specific summaries have explicit valid-case counts;
case-wise means are not pooled voxel Dice. Native validation selection and
canonical test absent-label handling remain intentionally distinct.
