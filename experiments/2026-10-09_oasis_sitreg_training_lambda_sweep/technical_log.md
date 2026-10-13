# SITReg training sweep technical log

## Known limitations

Four-GPU launch has tight startup memory on GPUs0/1, particularly GPU1;
per-process allocator caps cannot guarantee safety against later allocations
by unrelated jobs. Preserve any OOM failure without silently modifying the
scientific protocol. Runtime is diagnostic under shared workloads.
Native historical cuDNN benchmark mode is not bitwise deterministic.

## Scope and status

Approved four SITReg runs at lambda0.1,1/3,3,10 on GPUs0..3. Native200-epoch
protocol from P45, fresh initialization, no VFA runs or lambda1 retraining.
Production launched2026-10-09 14:29:43UTC (07:29:43PDT), after the reference,
validation, update and serialization gates below passed.

## Reference reproduction audit

Preflight attempt1 failed the initial1e-4 maximum field-error check:maximum
0.117886voxel, RMS0.001818voxel. No production run was launched. Five backend
diagnostics showed historical benchmark mode is non-bitwise reproducible even
within one process (unchanged-input repeats differed by0.005482voxel maximum).
Changing TF32/backend settings enlarged or changed the discrepancies, so the
historical backend was retained rather than selecting a new numerical mode.

Historical/current versions match:torch2.6.0+cu118, cuDNN9.1.0.70,
composable-mapping1.0.0, deformation-inversion-layer1.1.2. Comparing SITReg
against archived commit8332a5a (February2025) showed only later optional
feature-enhancement/combination branches; the default branch retains the
original computation. Numerical and metric reproduction, not bitwise equality,
is the valid gate under this benchmark-mode protocol. Revised bounds and an
additional full30-validation-pair check are specified in the execution spec.

## Accepted preflight and launch

Six CPU contract tests passed. Attempt2 passed strict checkpoint loading,
numerical rank0 field/metric reproduction, native30-pair validation and four
fresh one-step Adam/checkpoint roundtrips (all finite gradients).
Rank0 field RMS0.000386017voxel, max0.0947742voxel. Dice/HD95/ASD:
0.798720085/2.061319337/0.423048074 versus immutable reference
0.798719356/2.061319337/0.423064121. Full30 validation Dice0.814460264 versus
historical0.814463716. Initialization SHA256:
`64b709c6fa0f00042a86a315e55adc9c9a3e3d2cfded4746246bd53dffa67636`.
All45 source-file hashes are frozen by the gate and checked before every epoch.

Four-GPU detached controller launched after preflight exit. All four processes
are alive and all initial-weight hashes match the accepted preflight exactly.
Assignment:GPU0=lambda0.1, GPU1=1over3, GPU2=3, GPU3=10. Controller PID42779;
container worker PIDs42787/42788/42789/42790 respectively. All four workers
completed at least50 finite optimizer updates without failure or OOM. Their
first50 moving/fixed pair identities match exactly. GPUs0,2,3 also completed
their first validation/checkpoint cycle at the launch verification snapshot;
GPU1 was still in its first validation. Measured startup peaks on GPUs0,2,3
were33.33GiB allocated and35.94GiB reserved. These are launch checks, not final
training results; all runs continue toward200 epochs.
No VFA job was started. Container image ID:
`sha256:dab700c415830dcf07c16ef7f2901deed95bb104eda388514c3ce74a43354ba8`.
CPU thread count8/worker prevents unnecessary CPU oversubscription; native
GPU precision/model/loss/optimizer remain unchanged. Artifact store had15TiB
free at launch. Run manifests, launch.json, worker logs and progress.json
are external; no historical file or existing job is altered.

## Approved first100 detailed evaluation follow-up

User approved100-case evaluation for all four new checkpoints, starting the
completed lambda10 and queuing the other three after training. Added separate
evaluation/queue source files; all45 training-locked sources remain untouched.
All35 per-label rows per case will be saved, including explicit absent-label
exclusions, Dice/HD95/symmetric ASD and directional ASD. Reuse first10 field
bytes; infer ranks10–99. CPU contracts and GPU3 reused/fresh smoke are pending.
See first100_evaluation_spec.md and registry P49 for complete scope.

First100 smoke_v1 preserved a strict-check failure: saved rank0 accuracy and
foreground Jacobian metrics reproduced exactly, but float32 diffusion differed
by4.656612873e-10. NIfTI-loaded arrays have different strides than the original
inference buffers; numpy float32 summation order can change. Revised only
diffusion tolerance to1e-8; other metric tolerances remain1e-10. Added a
regression contract rejecting meaningful accuracy or diffusion changes.
Evaluator also restores C-contiguous array layout before metrics to match
the original inference buffer, without changing any field voxel values.
Smoke_v2 completed both cases and70 label rows with exact rank0 parity,
but final Markdown export failed under the container ASCII locale. Added
explicit UTF-8 report serialization and a regression test; no math changed.
The failed smoke is preserved. Measured inference peak14.38GiB allocated,
16.28GiB reserved; queue threshold remains24GiB free.

Smoke_v3 passed both full-volume cases, all70 label rows, exact saved-rank0
metric parity, native checkpoint inference, field serialization and UTF-8
report export. All15 combined training/evaluation contracts passed. No
training/native metric source changed. Queue launched2026-10-09 18:04PDT:
controller container PID57633; lambda10 worker PID57636 onGPU3. Lambda0.1,
1over3 and3 are waiting for their respective clean training+first10 exits.
Queue state and output tables are external under P49's artifact key.
Per-label details: per_label_per_case.csv/json; per-case means and wide label
columns: per_case.csv/json; individual cases have metrics.json and per_label.csv.

## Combined first100 report follow-up (2026-10-10)

All four evaluations completed with100 cases and3500 label rows each. The user
requested adding them to P48's four MR-D-BSCP weights and original VFA/SITReg
table. The separate `scripts/reports/oasis_first100_with_sitreg_sweep.py`
checks complete cohorts, case/direction identity, canonical evaluator hashes,
checkpoint/field hashes, per-label reconstruction and saved summary parity.
It recomputes equal-case means and sample SD from the existing canonical
per-case scores; no training or inference rerun is needed. Original reports
remain read-only. The two aggregation contract tests passed.

Combined export destination:
`${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_mrdbscp_vfa_sitreg_sweeps_comparison_v1/`.
Includes mean/SD CSV,1000 case rows, JSON provenance and Markdown. Original
SITReg lambda1 remains explicitly historical, not a new seed42 sweep run.

The requested2×3 accuracy–regularity figure is exported under the combined
report's `arc_2x3_v1/` directory as PNG/PDF/SVG, with plotted statistics and
checksum provenance. Columns show Dice/HD95/symmetric ASD against foreground
SD(log J); top row has mean curves only, bottom adds axis-aligned±1 across-case
sample-SD ellipses (not confidence or covariance ellipses). All plotted means
and SDs are reproduced from100-case rows. Historical SITReg lambda1 is a hollow
diamond with dashed adjacent segments. Two plotting tests and visual review
passed. No new inference or metrics were needed; source reports are unchanged.
