# Fixed-lambda1 prefix to dense-stage diffusion sweep

## Approved protocol

Seed3 shuffled OASIS ranks0–9, moving to fixed. Reuse the validated P43/P48
pyramid cps8/4/2 best-state prefixes at lambda1 for every stage. For each case,
run six FRESH dense refinements at lambda0.1/0.2/0.5/1/2/5, including a fresh
lambda1 fit. Prefixes, historical lambda1 dense fields and DL references stay
read-only. No GNCC or new image normalization is introduced.

Retain native local squared NCC9, unmasked total voxel-field forward-valid L2
diffusion, normalized XYZ parameters, raw stored intensities, border sampling,
align_cornersTrue and full-resolution dense loss. Exact detached prefix-field
clone; no cps1 controls or analytic B-spline regularizer. Fresh seed1 Adam,
betas0.9/0.999,eps1e-8,no AMSGrad/weight decay; LR0.001. Prefix LRs remain
0.03/0.01/0.003 with pyramid factors4/2/1; no prefix rerun is needed.

Dense maximum2000/minimum300 updates; median window25/check25/patience100;
relative total/data thresholds1e-4/5e-5. Halve LR on plateau at most twice,
at least100 updates between actions; stop on the third plateau. Save lowest
total-loss state, including initialization. Never choose checkpoints by labels.

## Execution and verification

Use the separate `fixed_prefix_dense.py` entrypoint; preserve all historical
numerical source files. Explicit CLI prefix/dense alpha values and stage-vector
run names. Validate legacy prefix spec/source/input/split/hash/DONE against
lambda1, independently of the selected dense lambda.

UID:GID1009:1020, cuDNN benchmarkFalse/deterministicTrue, TF32off, FP32.
One worker each on GPUs0/1/3; GPU2 and all other jobs untouched. Recheck free
memory before each launch; minimum4GiB or measured smoke tensor peak+2GiB,
whichever is larger. Preserve failures; one OOM retry after the slot is
released and sufficient memory is available. No silent protocol changes.

Tests cover60 unique configs, prefix identity, CLI/invalid values, unchanged
legacy validation, lambda1 value/gradient parity, loss weighting and complete
report generation. Rank0 one-step dense smokes at0.1 and5 use real production
prefixes and canonical DVF serialization/accuracy/regularity/native-loss checks.
Production only follows successful smokes. Require60 complete finite native
fields, per-case identical prefix hashes/stage records and checked best loss.

## Reports and artifacts

Run/report key under `${ARTIFACT_ROOT}/runs/` and `/repo_experiments/`:
`oasis_seed3_first10_mrdbscp842_prefixlambda1_dense_lambda_sweep_v1`.

Save per-case CSV/JSON, mean±sample-SD table and Markdown for six dense settings,
the unrefined prefix and original VFA/SITReg/FeDDA-1MF:100 case rows total.
Five metrics:Dice,HD95,symmetric ASD,foreground positive-J SD(log J),foreground
J≤0 percentage. Fixed raw-positive foreground; canonical absent-label rules.
Paired changes versus prefix and fresh lambda1, stopping/best epochs and runtime;
separate old/new lambda1 reproducibility appendix. Dice-first selection with
ASD thenHD95 tie breaks. Acceptance marker only after complete validation.
