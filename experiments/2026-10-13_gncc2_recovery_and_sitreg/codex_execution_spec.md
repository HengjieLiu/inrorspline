# P58 recovery and P59 SITReg GNCC²

User approved 2026-10-13: recover four interrupted FeDDA GNCC² runs, finish
all four training/evaluation jobs, then automatically train/evaluate SITReg
GNCC² at the same four lambdas. No other GPU jobs may be changed.

## Recovery audit

The original FeDDA controller and four workers correctly stopped on source
drift. The evaluator's only byte change was deletion of one final blank line.
Current SHA256 `e5507606b48f04da1195467a864e3d9d410e35872f7459b221e1010da5badfcd`
plus exactly one newline reproduces historical SHA256
`35504fab4e9174d6034f05174415433f46cffeaf0097877d4c8b080e8a2a2172`.
No numerical change, OOM or nonfinite checkpoint was identified.

| GPU | Lambda | Last saved epoch | Next epoch | Saved Adam updates |
|---:|---:|---:|---:|---:|
| 0 | 0.1 | 103 | 104 | 10300 |
| 1 | exact 1/3 | 120 | 121 | 12000 |
| 2 | 1 | 118 | 119 | 11800 |
| 3 | 3 | 120 | 121 | 12000 |

Never overwrite old failed directories, checkpoints, logs or the current
working evaluator. Freeze a source-only repository copy under the artifact
store, recover the historical evaluator bytes there, verify every file and
make files read-only. Historical repository-path hash checks resolve to this
verified copy. External model code, dependencies, data and references retain
their original paths and strict hash checks; they are not silently exempted.
Record frozen revision, per-file hashes, exact transformation and snapshot path.

Resume into distinct `_recovery_v1` training/evaluation/report directories.
Restore weights, optimizer including AMSGrad moments/steps, best validation
checkpoint and all RNG states. Reconstruct the persistent native DataLoader
with metadata-only dataset replay, including validation iterator seed draws.
Require all completed per-epoch pair hashes and CPU/Python/NumPy RNG states to
match saved evidence exactly. Preserve the actual sampler iterator at its
current position. Keep checking historical pair hashes after every new epoch.
Copy only history through the saved epoch; retain any later incomplete history
in the untouched failed directory. Runtime excludes downtime and includes the
original completed epochs. Historical cuDNN behavior remains non-bitwise.

## SITReg protocol

Start only after all four recovered FeDDA trainings, all 400 case evaluations
and its three-loss report pass. Then run a full native/reference preflight on
GPU 3, followed by four fresh jobs on GPUs 0/1/2/3 with lambdas
0.1/exact 1/3/1/3, respectively. Use UID:GID 1009:1020 throughout.

Only change signed GNCC to per-image `1-rho²`, with centered population
moments, FP32 variance-product floor, ordinary autograd and no similarity
rescaling. Native SITReg 15,076,110 parameters, seed 42, FP32 batch 1, joint
bidirectional average for similarity and unmasked total voxel L2 diffusion.
Adam AMSGrad LR 1e-4, 200 epochs x 50 pairs, one update/pair = 10,000 updates.
No scheduler or early stopping. Select checkpoints using native 30-pair
validation Dice only. Preserve initialization and every epoch's training-pair
hash against original LNCC controls.

Retain approved native CUDA override: benchmark True, deterministic False,
cuDNN TF32 True, matmul TF32 False. Training memory gate is the greater of
37 GiB and ceil(measured reserved peak + 1 GiB); preflight waits for 38 GiB.
FeDDA training/evaluation and SITReg evaluation retain their 24 GiB gate.

## Verification and outputs

- CPU formula/gradient, unchanged training semantics, iterator/RNG recovery at
  partial and exhausted sampler boundaries, unique names and all-four barrier.
- Four GPU checkpoint/optimizer/iterator validations before resumed updates.
- SITReg native checkpoint/field and full validation parity at unchanged
  tolerances; four fresh training smokes, trained-weight gradient smoke and
  ranks 0/10 serialization with 35 canonical label records each.
- Detailed 100-case inference per lambda; 3 accuracy + 2 foreground regularity
  metrics, 3500 label records per run, absent-label exclusions, saved fields,
  warped labels, all source/checkpoint hashes and completion markers.
- FeDDA: existing strict three-loss reporter, new recovery report directory.
- SITReg: strict new-vs-LNCC component plus accepted signed GNCC references;
  13 settings, 1300 case records and 2x3 three-loss stdlogJ ARC. Means top,
  marginal sample-SD ellipses bottom; historical LNCC lambda1 clearly marked.
- Persistent queue records commands, PIDs, dependencies and failures. Failures
  block downstream launches; no automatic tolerance relaxation or retraining.

Artifact keys: `oasis_gncc2_recovery_then_sitreg_v1`,
`oasis_gncc2_recovery_source_snapshot_v1`,
`oasis_fedda_gncc2_diffusion_training_lambda_sweep_v1_recovery_v1`,
`oasis_seed3_first100_fedda_gncc2_training_lambda_eval_v1_recovery_v1`,
`oasis_seed3_first100_fedda_three_ncc_v1_recovery_v1`,
`oasis_sitreg_gncc2_training_preflight_v1`,
`oasis_sitreg_gncc2_diffusion_training_lambda_sweep_v1`,
`oasis_seed3_first100_sitreg_gncc2_training_lambda_eval_v1`.
