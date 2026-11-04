# Cps8/4/2 weights [1,1,5] followed by dense lambda sweep

## Approved scope

Use the ten accepted P61 cps2lambda5 outputs, seed3 shuffled OASIS ranks0–9.
Their frozen cps8/cps4 weights remain1; cps2 is5. Run70 fresh dense fits at
lambda0.1/0.2/0.5/1/2/5/10. No prefix rerun, cps1 controls, or historical
artifact/source mutation. Stage vectors are [1,1,5,dense_alpha].

## Numerical settings

Exact detached clone of each saved full-resolution normalized XYZ field;
reconstruct from saved cps2 controls as an independent initialization check.
Use unchanged raw stored intensities, local squared NCC9, border sampling,
align_cornersTrue, and unmasked total-field voxel forward-valid L2 diffusion.
No loss normalization, global NCC, feature loss, label loss, or DVF supervision.
Dense LR0.001, fresh seed1 Adam, FP32, cuDNN benchmarkFalse/deterministicTrue,
TF32off. Max/min2000/300, median/check25, patience100, relative total/data
thresholds1e-4/5e-5; up to two LR halvings with100 epochs between actions.
Select the lowest total-loss state including initialization, not label metrics.

## Execution and validation

New runner: scripts/experiments/oasis_alpha_sweep/cps2lambda5_dense.py.
Run/report key beneath external ${ARTIFACT_ROOT}/runs and /repo_experiments:
oasis_seed3_first10_mrdbscp842_prefix115_dense_lambda_sweep_v1.
One worker each on GPUs0/1/3, UID:GID1009:1020, existing container image/digest
recorded. LeaveGPU2 and all existing jobs untouched. Require4GiB free or smoke
tensor peak+2GiB, whichever is larger. Preserve failures; at most one OOM retry.

Validate P61 acceptance/report hashes, numerical sources, exact case/split/input
identity, stage weights, checkpoints, finite native fields and completion.
Test70 unique names, fixed source prefix, CLI invalid values, legacy loss and
gradient parity, and complete reporting. Gate production on rank0 one-step
lambda0.1/10 smokes through serialization and canonical metrics. Require all70
complete finite [160,224,192,3] fields, identical per-case prefix hashes/stage
records, exact cloning and best-checkpoint loss consistency before aggregation.

## Report

CSV/JSON/Markdown: seven settings plus unrefined [1,1,5] prefix and original
VFA/SITReg/FeDDA,110 case rows,11 complete ten-case groups. Dice,HD95,symmetric
ASD, fixed-raw-positive foreground positive-J SDlogJ and J<=0 folding percent;
mean ± sample SD. Paired deltas versus prefix and fresh dense1. Include stage
epochs, runtime and GPU. Dice-first selection breaks ties by ASD thenHD95.

Also compare against P60 [1,1,1] prefix dense results at the six shared weights
(60 paired records). No prior dense10 result exists; do not fabricate one or
describe this changed-initialization comparison as a reproducibility test.
All first10 results are exploratory test-case tuning; shared-load timing is
diagnostic. Write acceptance only after full-cohort and report validation.
