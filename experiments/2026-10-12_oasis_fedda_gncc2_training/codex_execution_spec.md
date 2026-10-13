# FeDDA squared global NCC sweep plan

Status: approved for implementation and execution on 2026-10-12. Run preflight
gates before launching four trainings, dependent evaluations, and the report.

## Objective and scope

Train four fresh original FeDDA-1MF models using `1 - GNCC**2` plus the same
native L2 diffusion used by the completed signed-GNCC sweep. Evaluate each
best-validation checkpoint on the same seed-3 shuffled first 100 OASIS test
pairs. Compare squared GNCC, signed GNCC, and squared local NCC in one
six-panel alignment-regularity figure.

The [loss normalization note](../../docs/implementation/ncc_loss_normalization.md)
defines the formulas, stabilization, support, and limits of equal-lambda
comparisons. No calibration multiplier or automatic lambda extension is part
of this experiment.

## Training matrix and execution

| Physical GPU | Lambda key | Actual lambda | New training runs |
| --- | --- | --- | --- |
| 0 | `0.1` | 0.1 | 1 |
| 1 | `1over3` | 1/3, not 0.3 | 1 |
| 2 | `1` | 1 | 1 |
| 3 | `3` | 3 | 1 |

Run four independent processes, one added worker per GPU, with explicit UUID
pinning and visible CUDA device 0. Each worker runs training and its dependent
100-case evaluation sequentially on that GPU. Container UID:GID is 1009:1020,
matching the approved FeDDA runs. Preserve all other GPU workloads.

After approval, measure current free memory/utilization and a new preflight
peak. Use the larger of 24 GiB and `ceil(peak_reserved_GiB + 2)` as launch
headroom. The prior signed-GNCC peak is context, not a current memory reading.
If a slot lacks headroom, wait without changing existing jobs. Preserve OOM
or other failures and stop that candidate for diagnosis; do not automatically
change precision, batch size, LR, backend, or retry until a gate happens to pass.

## Frozen native protocol

| Setting | Required value |
| --- | --- |
| Model | Original `exp_midl_0407/fedda_1mf`, 2,854,079 parameters, `use_batch_parallel=False` |
| Initialization | Fresh seed 42; no warm start from a trained GNCC or LNCC model |
| Data | Same native 300 training subjects and 30 validation pairs |
| Geometry | Native LIA-to-RAS, 160 x 224 x 192, 1 mm |
| Intensities | Native stored intensities; no added normalization or mask |
| Loader | Batch 1, workers 0, pinned memory, persistent shuffled iterator |
| Budget | 200 epochs, 50 pairs per epoch |
| Updates | Two sequential forward then reverse updates per pair; 20,000 Adam updates total |
| Optimizer | Adam AMSGrad, constant LR 1e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay 0 |
| Schedule | No LR scheduler or early stopping |
| Precision | FP32, no AMP |
| Image sampling | Native border padding and `align_corners=True` |
| Validation | Native forward-only validation after every epoch |
| Checkpoint selection | Highest full-precision native validation Dice; keep the earlier checkpoint on an exact tie |

The reverse update must use the already updated model weights. Do not replace
this with SITReg's single jointly averaged bidirectional update. Match the
initial-state hash and each epoch's pair-stream hash to the prior FeDDA controls.

Explicit historical backend override: `cudnn.benchmark=True`,
`cudnn.deterministic=False`, cuDNN TF32 enabled, matmul TF32 disabled. This
preserves the comparison protocol instead of silently adopting the repository's
new-registration default. Record requested and effective flags; bitwise
cross-process training reproducibility is not claimed.

For an overall lambda recommendation, use the highest native validation Dice
among the four selected checkpoints; exact ties use lower lambda. Test-set
metrics do not select checkpoints, epochs, or the recommended lambda.

## Only change the similarity term

Implement a new stateless per-image population GNCC loss:

1. Require equal floating `[B,1,D,H,W]` fixed and warped tensors.
2. Recompute both images' centered population moments over `(C,D,H,W)` for
   every batch element, every call, and each training direction.
3. Compute `rho = covariance / sqrt(clamp_min(var_fixed * var_warped, eps))`,
   with `eps = 1.1920928955078125e-7` for this FP32 protocol.
4. Return `mean_batch(1 - rho.square())`. Square before batch reduction.
5. Use ordinary autograd and the same native `Grad3d(penalty='l2')` on the
   unmasked total voxel-unit displacement: `total = similarity + lambda * diffusion`.

No local windows, sign-restored squared score, masks, features, segmentation
loss, displacement supervision, sample-standard-deviation normalization,
custom backward, detached rho, initial-loss scaling, or extra factor of 0.5.
Do not transplant the cached-fixed-image dense-fitting GNCC class unchanged.
Record similarity, unweighted diffusion, weighted diffusion, total loss, and
signed rho diagnostics without consuming training RNG or changing pair order.

Use separate adapters and entry points so old numerical code and hash-locked
outputs remain unchanged. Proposed source files after approval:

- `scripts/experiments/oasis_alpha_sweep/fedda_gncc2_training.py`
- `scripts/experiments/oasis_alpha_sweep/run_oasis_fedda_gncc2_training.py`
- `scripts/experiments/oasis_alpha_sweep/run_oasis_fedda_gncc2_eval.py`
- `scripts/experiments/oasis_alpha_sweep/launch_oasis_fedda_gncc2.py`
- `scripts/reports/oasis_fedda_three_ncc_report.py`
- `tests/test_oasis_fedda_gncc2.py`

Use distinct loss identity `global_ncc_squared_population`, protocol `1-rho^2`,
and run names `fedda_1mf_gncc2_lambda_{key}_seed42`. Reject mixed signed/squared
checkpoints, wrong lambda metadata, or attempts to overwrite prior directories.

## Preflight and acceptance gates

Before production training, after implementation approval:

1. Verify historical source/checkpoint/dataset hashes and accepted GNCC/LNCC
   reference reports. Load the original FeDDA checkpoint strictly and run
   reference inference, native validation, and canonical metric parity with
   unchanged historical tolerances. Investigate nondeterministic discrepancies;
   preserve failed evidence rather than relax a gate.
2. Unit-test direct population-formula parity and autograd gradients, positive
   and negative correlation, constant/low-variance inputs, the variance floor,
   batch-element independence, stateless calls, and `mean(1-rho**2)` rather than
   `1-mean(rho)**2`. Check the `2*rho` gradient identity against signed GNCC.
3. Test exact lambda parsing, unique names, protocol guards, all four GPU
   indices, protected historical outputs, and unchanged old-runner behavior.
4. On GPU 3, sequentially smoke all four lambdas through fresh forward/reverse
   Adam updates, save/reload, and continuation. Also test a trained-checkpoint
   update and repeated warm steps for peak-memory measurement. Verify only
   the intended model parameters update and the optimizer step count is two
   per training pair.
5. Smoke canonical ranks 0 and 10 through finite DVF serialization and all
   35 label rows per case. Confirm rank 0 is fixed 0333, moving 0115.
6. Require completion markers for all four trainings and evaluations, 400
   finite correctly shaped fields, and 14,000 new per-label records. Validate
   100 unique ordered ranks and identities for every setting. Reconstruct
   case means from included labels and aggregate statistics from case records.

Preflight does not authorize changes to prior sources or outputs. Any material
protocol mismatch must be resolved before launching the four training jobs.

## Detailed test evaluation

Use ranks 0-99 of
`configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`, moving to fixed.
Export the selected checkpoint's first 10 fields, retain those exact bytes,
and reuse them when completing the detailed first 100 evaluation. Infer only
ranks 10-99 anew for that extension. No test-case tuning occurs in this sweep.

Save fields as finite `[3,160,224,192]` channel-first zyx voxel displacements,
with case identity, geometry, checkpoint hash, field hash, and direction.
For every case save all labels 1-35, including excluded labels:

- Dice, canonical area-weighted HD95 in mm, pooled-area symmetric ASD in mm,
  and both directional ASDs.
- Fixed, original moving, and warped label voxel counts; explicit inclusion
  flags and exclusion reasons. Follow the canonical exclusion if a label is
  absent in any of these three segmentations. Serialize excluded values as
  null/blank, not zero; never silently change the valid-label denominator.
- Foreground std(log J) on positive Jacobians and foreground `J <= 0` folding
  percentage, using fixed raw image > 0 and the canonical two-voxel crop.
- Full-volume diffusion and foreground diffusion; the foreground version
  uses forward-valid edges with both endpoints in fixed raw image > 0, no
  extra crop, and the same axis/component means as native diffusion.
- Validation-selected epoch, runtime, GPU assignment, and inference metadata.

The headline cohort is all 100 pairs; provide first-10 summaries separately.
Use equal case weights and sample SD (`ddof=1`) across cases, not pooled label
statistics. Preserve the full per-case/per-segmentation accuracy records.

## Three-loss report and six-panel ARC figure

Reuse immutable accepted references:

| FeDDA similarity | Lambdas | Case records |
| --- | --- | --- |
| New `1 - GNCC^2` | 0.1, exact 1/3, 1, 3 | 400 |
| Existing `1 - GNCC` | 0.1, exact 1/3, 1, 3 | 400 |
| Existing `1 - LNCC^2` with window 9 | 0.1, exact 1/3, 1, 3, 10 | 500 |

The final report contains 13 operating points and 1,300 matched case records.
Retain LNCC lambda 10 as an existing range-context endpoint; do not train
GNCC-squared lambda 10 or extrapolate either global curve. Mark historical
LNCC lambda 1 with its existing hollow symbol and dashed adjoining segments;
it is not a fresh matched-seed control.

Primary figure: `fedda_three_ncc_stdlogj_arc_2x3`, exported as PNG, PDF, SVG.

| Row | Column 1 | Column 2 | Column 3 |
| --- | --- | --- | --- |
| Top | Mean Dice vs foreground std(log J) | Mean HD95 vs foreground std(log J) | Mean symmetric ASD vs foreground std(log J) |
| Bottom | Same curve plus SD ellipses | Same curve plus SD ellipses | Same curve plus SD ellipses |

All six panels show all three loss curves with consistent colors, distinct
markers, lambda labels, and common x limits. Pair y limits within columns.
Join points in lambda order, with no fitted smooth curve. Top-row curves have
no uncertainty ellipses. Bottom-row ellipses have horizontal and vertical
semiaxes equal to the respective marginal sample SD across the 100 cases.
They are axis-aligned descriptive spreads, not covariance ellipses, confidence
intervals, or uncertainty across training seeds.

Provide CSV, JSON, and Markdown with means/SDs for all three accuracy metrics,
both Jacobian regularity metrics, and foreground/full diffusion. Save paired
per-case differences for each loss comparison at the four common lambdas,
plus validation histories, selected epochs, and runtime diagnostics. Explain
equal-lambda results separately from comparisons at overlapping achieved
regularity; do not extrapolate or construct hybrid points from different fields.
The requested deliverable is this std(log J) figure; diffusion/folding data
remain available for additional plots without requiring new training.

Use the accepted P57 report and foreground-metric source records as read-only
inputs. Record their hashes in the new report. The three-loss figure requires
a new reporter because the current two-loss reporter deliberately validates
exactly nine settings. Do not weaken that historical validator.

## Provenance and output locations

Record source revision plus dirty-source hashes, model/config/data/pair hashes,
seed/RNG, initial-state and epoch pair-stream hashes, full optimizer settings,
requested/effective backend flags, Docker image identity, device UUID and live
memory/utilization, loss histories, best/latest checkpoints, and failure logs.

New artifact roots, created only after execution approval:

- `${ARTIFACT_ROOT}/runs/oasis_fedda_gncc2_diffusion_training_lambda_sweep_v1`
- `${ARTIFACT_ROOT}/runs/oasis_seed3_first100_fedda_gncc2_training_lambda_eval_v1`
- `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_fedda_three_ncc_v1`

Aggregate the accepted final report only after all gates pass. Label any
interim results incomplete. Historical source edits, reference replacement,
test-driven range extension, or additional seeds are not authorized.
