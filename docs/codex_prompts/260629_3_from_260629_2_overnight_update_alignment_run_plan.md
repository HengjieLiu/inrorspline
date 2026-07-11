# Overnight Update-Alignment Run Plan

Source-linked adaptation:

- Primary source prompt:
  `docs/codex_prompts/260629_2_smaller_plan.md`
- Technical source prompt used as optional extension context:
  `docs/codex_prompts/260629_1_sinr_cp_kernel_experiment_plan.md`
- Adaptation date: 2026-06-29
- User-approved defaults: run the first three diagnostic cases first, use GPUs
  `0,1,2,3`, and keep the original two source prompts unchanged.

## Objective

Create an overnight, self-supervised experiment and report workflow that tests
whether image-based optimization updates point toward strong teacher
deformations, while keeping the paper footprint compact.

The immediate scientific question is:

```text
Can SINR and direct B-spline CP methods represent teacher fields, but still
produce less useful early image-registration updates under the task's actual
image loss and motion regime?
```

This adapted run is intentionally smaller and more execution-oriented than the
source prompt. It should produce a review notebook, summary tables, and one
main diagnostic figure before expanding to more cases.

## Cases To Run First

Run exactly these three cases first:

```text
brain/oasis:
  case: seed-3 shuffled OASIS rank 9
  expected sorted index: 88
  fixed/moving: 0173 <- 0136
  teacher field: saved BS4 SITReg dense displacement
  analysis mask: fixed raw image-positive foreground, fixed_image > 0

lungct:
  case: DIR-Lab 4DCT case 01
  teacher field: saved pTVReg displacement reconstructed on canonical pTV 1mm crop
  expected pTVReg snap TRE: 0.7671410824 mm
  expected pTVReg TRE std: 0.8917488134 mm
  analysis mask: fixed T00 lung mask on canonical pTV 1mm crop

lungct:
  case: DIR-Lab 4DCT case 08
  teacher field: saved pTVReg displacement reconstructed on canonical pTV 1mm crop
  expected pTVReg snap TRE: 1.0142198317 mm
  expected pTVReg TRE std: 1.2922227355 mm
  analysis mask: fixed T00 lung mask on canonical pTV 1mm crop
```

Do not expand beyond these three cases until the overnight run completes and the
report validates cleanly.

## Methods

### OASIS Rank 9

Use the completed first10 operating points:

```text
idir_original:
  alpha: 1
  lr: 1e-4
  epochs: 2500
  regularizer: original IDIR coordinate-autograd BE scale

sinr_cps4_be_bspline_analytic:
  alpha: 1000
  lr: 1e-4
  epochs: 2500
  regularizer: cubic B-spline analytic BE

direct_cp_cps4_be_bspline_analytic:
  alpha: 1000
  lr: 1e-3
  epochs: 2500
  regularizer: cubic B-spline analytic BE
```

### LungCT Case 01 And Case 08

Use the pTV-preprocessed registration domain and completed default operating
points:

```text
idir_be_default_lungmask:
  alpha_bending: 10
  lr: 1e-4
  epochs: 2500
  training domain: fixed/T00 lung-mask sampled coordinates

sinr_cps4_ptv_lcc_isotv:
  ptv_isotv_weight: 0.11
  lr: 1e-4
  epochs: 2500
  loss: pTV-style local CC, border_mask=5, isoTV

direct_cp_cps4_ptv_lcc_isotv:
  ptv_isotv_weight: 0.11
  lr: 1e-3
  epochs: 2500
  loss: pTV-style local CC, border_mask=5, isoTV

direct_cp_32_16_8_4_ptv_lcc_isotv:
  ptv_isotv_weight: 0.11
  stage_full_cps: 32,16,8,4
  stage_downsample: 8,4,2,1
  stage_epochs: 200,400,800,1000
  stage_lr: 1e-3
  purpose: multistage capture-range reference
```

## Output Roots

Use large-output storage for run artifacts:

```text
container run root:
  /database/inrorspline/runs/update_alignment_overnight_20260629_v1

repo review root:
  experiments/2026-06-29_update_alignment_overnight
```

Expected review artifacts:

```text
experiments/2026-06-29_update_alignment_overnight/
  codex_execution_spec.md
  technical_log.md
  update_alignment_review.ipynb
  update_alignment_report.md
  tables/
    checkpoint_alignment_rows.csv
    case_summary_rows.csv
    method_summary_rows.csv
    teacher_fit_reuse_rows.csv
  plots/
    brain_lung_alignment_curves.png
    early_alignment_summary.png
    sinr_directcp_update_similarity.png
```

Large checkpoint samples, dense final fields, job logs, and model artifacts stay
under `/database/inrorspline/runs/update_alignment_overnight_20260629_v1`, not
under Git.

## Self-Supervised Workflow

Implement a resumable launcher with explicit phase gates. Suggested script:

```text
scripts/experiments/update_alignment/launch_update_alignment_overnight.py
```

Required launcher features:

```text
--dry-run
--smoke
--run-full
--resume
--gpus 0,1,2,3
--scope first3
--out-root /database/inrorspline/runs/update_alignment_overnight_20260629_v1
--review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_update_alignment_overnight
```

The launcher should write:

```text
phase_status.json
job_manifest.csv
job_manifest.json
launcher_stdout.log
launcher_stderr.log
```

State model:

```text
planned -> validated -> smoke_passed -> running -> aggregated -> reported
failed_validation | failed_smoke | failed_run | failed_aggregation
```

Completion markers:

```text
RUNNING
DONE
FAILED
summary.json
checkpoint_samples.csv or checkpoint_samples.npz
```

Retry policy:

- Retry at most once for transient process failures such as an interrupted
  worker or CUDA context error.
- Do not retry deterministic validation failures, missing teacher files,
  pTVReg TRE validation mismatches, nonfinite losses, or repeated OOM.
- If one method/case fails, finish already-running jobs, then stop before
  aggregation unless `--allow-partial` is explicitly passed.

Watchdog:

- Poll child jobs every 300 seconds.
- Record GPU assignment, PID, start time, latest log mtime, and latest parsed
  progress row.
- Stop if no progress is observed for 45 minutes on an active job.

## Checkpoint Field Logging

Do not save full dense LungCT fields at every checkpoint. Use deterministic
sampled field logging plus final dense fields.

Sampling rules:

```text
sample_seed: 1729
oasis_sample_domain: fixed_image > 0
lungct_sample_domain: fixed T00 lung mask on pTV 1mm crop
max_sample_points_per_case: 100000
sample_file: sample_points_<dataset>_<case>.npz
```

Checkpoint epochs for single-stage methods:

```text
1,10,25,50,100,250,500,1000,1500,2000,2500
```

For multistage LungCT DirectCP, save both global epoch and stage identifiers:

```text
stage_index
stage_cps
stage_downsample
stage_epoch
global_epoch
```

Save sampled displacement values in common analysis units:

```text
oasis: voxel units in fixed OASIS grid
lungct: reg1mm voxel units on canonical pTV crop
```

Also save enough metadata to reconstruct source units:

```text
source_units
field_direction
component_order
grid_shape_zyx
mask_definition
teacher_source
```

## Metrics

For every consecutive checkpoint interval:

```text
delta_u_t = U_{t_next} - U_t
teacher_residual_t = U_teacher - U_t
update_alignment = cos(delta_u_t, teacher_residual_t)
update_norm
teacher_residual_norm
teacher_distance_before
teacher_distance_after
teacher_distance_delta
```

For each method/case:

```text
early_alignment_mean
early_alignment_median
early_alignment_auc
cumulative_early_alignment
final_teacher_distance
final_registration_metric
```

Early interval definition:

```text
single-stage: checkpoints up to epoch 250 by default
multistage: first stage plus first checkpoint of second stage; report stage-aware values
```

SINR vs DirectCP similarity:

```text
cos(delta_u_sinr, delta_u_directcp)
cos(cumulative_delta_u_sinr_early, cumulative_delta_u_directcp_early)
```

Final registration metrics:

```text
oasis: Dice, HD95, ASD, foreground regularity
lungct: original_snap TRE as primary, plus original_subvoxel, reg1mm_snap, reg1mm_subvoxel
```

Teacher fitting evidence:

- Reuse completed OASIS rank-9 SITReg field-fit results where compatible.
- Reuse completed LungCT case01/case08 pTVReg-field fitting results where
  compatible.
- If reuse validation fails, report the missing compatibility instead of
  silently rerunning.

## Preflight Gates

Before any full run:

1. Add or update the experiment registry entry.
2. Create `experiments/2026-06-29_update_alignment_overnight/`.
3. Add `codex_execution_spec.md`.
4. Add `technical_log.md` with limitations near the top.
5. Confirm Docker container `py310sitreg_v1_test1` is running.
6. Confirm GPUs `0,1,2,3` are visible.
7. Confirm output root is writable and has sufficient free space.
8. Validate OASIS rank-9 fixed/moving IDs.
9. Validate LungCT case01 and case08 pTVReg teacher TREs against saved `.mat`
   files.
10. Compile changed scripts inside Docker.
11. Run smoke jobs.

Suggested preflight command shape after implementation:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/launch_update_alignment_overnight.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --dry-run \
  --out-root /database/inrorspline/runs/update_alignment_overnight_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_update_alignment_overnight
```

Suggested smoke command shape after implementation:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/launch_update_alignment_overnight.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --smoke \
  --resume \
  --out-root /database/inrorspline/runs/update_alignment_overnight_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_update_alignment_overnight
```

Suggested full overnight command shape after smoke passes:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/launch_update_alignment_overnight.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --run-full \
  --resume \
  --out-root /database/inrorspline/runs/update_alignment_overnight_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_update_alignment_overnight
```

## Acceptance Criteria

Smoke acceptance:

- OASIS rank-9 case validation passes.
- LungCT case01 and case08 pTVReg teacher validation passes.
- One short OASIS job writes finite sampled checkpoint fields.
- One short LungCT job writes finite sampled checkpoint fields.
- Aggregator can compute at least one finite update-alignment row.

Full-run acceptance:

- Expected OASIS jobs complete: `3/3`.
- Expected LungCT jobs complete: `8/8` because there are 4 methods for each of
  2 LungCT cases.
- All accepted jobs have `DONE`, `summary.json`, and checkpoint sample outputs.
- Alignment values are finite and bounded by `[-1,1]` up to numerical epsilon.
- Teacher residual norms are nonzero at the first checkpoint.
- LungCT report keeps `original_snap` as the headline TRE.
- Review notebook executes in Docker and renders tables/plots inline.
- Large fields/checkpoints remain outside Git.

## Report Structure

The overnight report should answer these questions:

1. Does teacher-field fitting evidence still support "expressivity is not the
   bottleneck" for the selected cases?
2. Which method has the best early update alignment on OASIS rank 9?
3. Which method has the best early update alignment on LungCT case01 and case08?
4. Does LungCT multistage DirectCP improve early/cumulative update alignment
   relative to single-stage spline methods?
5. Are SINR and DirectCP sampled updates directionally similar?
6. Are any conclusions blocked by sign, unit, mask, or protocol mismatches?

Use cautious paper language:

```text
These selected-case diagnostics separate representation capacity from the
trajectory induced by image-based optimization. They do not prove a global
operator-level equivalence between methods.
```

## Expansion Gate

Only after the first three cases pass:

- Consider OASIS ranks `0-9` using the same seed-3 shuffled protocol.
- Consider LungCT cases `01-10`.
- Consider the optional SINR CP-kernel follow-up adapted in
  `docs/codex_prompts/260629_4_from_260629_1_optional_sinr_cp_kernel_followup_plan.md`.
