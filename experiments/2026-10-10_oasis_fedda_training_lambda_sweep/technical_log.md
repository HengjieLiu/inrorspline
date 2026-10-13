# FeDDA training sweep technical log

## Known limitations

Historical lambda1 source is not an immutable April2025 snapshot. Native
checkpoint inference/validation parity gates current adapters before training.
Historical CUDA benchmark mode is explicitly retained; seed42 does not ensure
bitwise CUDA training. Shared-GPU timing is diagnostic. GPU0 historical21.127GiB
reserved peak versus current38.6GiB free suggests feasibility,not a guarantee.

## Approved protocol

Original FeDDA-1MF200-epoch training,not later101-epoch onepass tune_1ncc.
Placement:GPU0 lambda3 then10;GPU1 lambda0.1 afterVFA;GPU2 lambda1/3 afterVFA.
Each dependency includes detailed first100 evaluation. GPU3 untouched.
Implementation and diagnostic qualification completed; see launch record below.

## Read-only baseline audit

Original checkpoint dice0.8141_epoch0196.pth.tar contains19600 Adam steps,
LR1e-4,AMSGrad,betas0.9/0.999,eps1e-8. Native source confirms50 pairs and100
sequential directional updates per epoch.2854079 trainable parameters.
Historical200-epoch training+validation time5.73252h (not current ETA).
The prior turn recalculated100 saved fields on CPU without writing artifacts:
mean Dice0.8251776226,HD95mm1.8358151722,ASDmm0.3595527842,
foreground SDlogJ0.6981786867,folding1.0480311287%. These will be persisted with
field provenance by the approved baseline/report stage; do not treat chat-only
summary values as a substitute for saved case records.

## Native benchmark-mode reproduction diagnosis

Nine CPU contracts passed. Initial GPU0 preflight is preserved under
`oasis_fedda_ncc9_diffusion_training_preflight_v1/`: strict checkpoint/parameter
loading succeeded,but the copied VFA field gate(max0.001,RMS0.0001voxel) failed
with max0.0106673,RMS0.000124745. No production was launched.

Bounded audit `oasis_fedda_reference_backend_audit_v1/` compared repeated
inference with unchanged weights and inputs. A fresh native benchmark process
gave max0.0527278,RMS0.000362093 versus the saved field; its immediate repeat
was bitwise identical. A third native process gave max0.00156474,RMS0.0000222516.
This cross-process variation is consistent with cuDNN algorithm selection.
Disabling TF32 increased discrepancy to max0.144563,RMS0.000737185; deterministic
TF32 gave max0.0440872,RMS0.000327071. Native benchmark/TF32 settings are retained.

The third native run's rank0 Dice delta+0.00000842887,HD95 delta0,ASD
delta-0.0000149624mm passed the original accuracy gates. Foreground folding was
identical; SDlogJ delta-0.0000192970. Full30 validation Dice0.8141128583 differs
from checkpoint0.8141095882 by+0.00000327006,well inside1e-4.

After diagnosis,FeDDA-specific field bounds are max0.1voxel/RMS0.001voxel;
original accuracy/validation bounds unchanged,plus explicit regularity bounds.
This gate refinement is not an optimization/backend/precision change and does
not claim bitwise inference or prove all historical source is identical.
Second preflight will use a distinct directory; initial failure remains intact.

Historical first100 evaluation completed and was saved with3500 per-label rows
and source/input/field hashes in the external report root's historical_lambda1/.
All five cohort means exactly reproduce the preceding read-only audit.

## Completed diagnostic and explicit qualification

The second preflight, preserved as
`oasis_fedda_ncc9_diffusion_training_preflight_v1_attempt2/`, had maximum field
error 0.0154247284 voxels and RMS 0.0001385311, but failed the 0.001 mm HD95
gate. That version did not save actual metrics before raising; its exact HD95
delta is unavailable. The runner now saves both metric sets before checking.
No claim is made that this failed HD95 check passed or was immaterial.

The bounded `oasis_fedda_metric_memory_diagnostic_v1/` audit then completed with
zero issues. Its rank-0 field was **bitwise identical** to the saved original:
Dice 0.7965444746455527, HD95 2.056167312275727 mm, ASD
0.43645652624264236 mm, foreground SDlogJ 0.7632773442609185, folding
1.602228735296796%. Native 30-pair validation Dice was 0.814107276487874,
versus checkpoint 0.814109588187874 (difference -0.0000023117).

All four lambda smokes completed two sequential Adam updates with finite
gradients and exact checkpoint roundtrips. Cold peak allocated memory was
18.96224 GiB. Maximum reserved memory was 21.22461 GiB, including six warm
updates. Production retains the 24 GiB free-memory gate.

The audit deliberately has no PASSED marker. A separate CPU qualification
script validates its stored evidence without another stochastic inference:
bitwise field equality, exact canonical metrics, validation tolerance,
four lambda/update/checkpoint contracts, original backend/UID/GID, memory and
all source/input hashes. It creates a new certificate and explicitly links
both prior failures without modifying them. No accuracy tolerance is relaxed.
This certifies **adapter correctness**, not cross-process repeatability; the
historical autotuned backend remains nondeterministic. The exact field result
does not erase the previously observed HD95 variation or prove an immutable
historical source snapshot. No training configuration is selected using these
test-case reproduction checks.

The detailed baseline is already persisted: 100 cases and 3500 per-label rows,
with source/checkpoint/data/field hashes. Twelve CPU contracts cover training,
queues, reporting, audit certification and rejection of failed/approximate
audits. Production launch status is recorded separately below.

## Production launch — 2026-10-10 02:40 PDT

Twelve CPU tests passed. The independent CPU certificate was saved as
`oasis_fedda_ncc9_diffusion_training_preflight_certified_v1/`, preserving both
prior failures and the read-only diagnostic. Source hashes remain frozen.
Controller container PID 69531 launched lambda3 on physical GPU0 (worker
PID 69539) as UID:GID 1009:1020. At launch GPU0 had 39513 MiB free and 55%
utilization from existing work. No existing process or queue was modified.
At02:42PDT, epoch1 had completed at least22 Adam updates with finite losses;
the model's seed42 initialization hash matched the certified audit. Startup
and first native training updates succeeded without an OOM.

The controller queued lambda10 behind FeDDA3 training plus detailed100 eval;
lambda0.1 and1/3 wait for the corresponding VFA training plus detailed100 eval
on GPUs1/2. GPU3's VFA queue is unchanged. Each FeDDA run uses200 epochs and
automatically evaluates its best-validation checkpoint on100 shuffled test
pairs, preserving all3500 per-label records. Final report generation starts
only after all four clean training/evaluation completions. Failed runs remain
preserved and are not automatically retried.

Run root: `oasis_fedda_ncc9_diffusion_training_lambda_sweep_v1/` under the
external artifact store's `runs/`. Historical baseline and final comparisons:
`repo_experiments/oasis_seed3_first100_fedda_training_lambda_sweep_v1/`.

## Four-method ARC export

After both VFA and FeDDA queues reported complete, the user requested adding
their sweeps to the existing MR-D-BSCP/SITReg2×3 figure. The separate report
script `scripts/reports/oasis_first100_four_method_arc.py` validates complete
cohorts, canonical metric conventions/source identity, source report hashes,
VFA checkpoint provenance and means/sample SD reconstructed from case rows.
The export includes19 settings and1900 paired case records: MR-D-BSCP four
weights and five points each for SITReg/VFA/original200-epoch FeDDA-1MF.
Each DL lambda1 point is historical, not a new seed42 retraining; hollow
diamonds and dashed adjacent segments retain that distinction.

The2×3 layout has Dice/HD95/symmetric ASD versus foreground SD(log J), mean
curves on top and axis-aligned±1 sample-SD ellipses below. These ellipses are
not confidence or covariance regions. No new inference, metric evaluation,
training or changes to existing outputs were needed. Two plotting/cohort
tests and visual inspection passed. PNG/PDF/SVG, all plotted mean/SD values,
case rows and checksum manifest are under
`${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_four_method_sweeps_arc_v1/`.
