---
doc_type: codex_execution_spec
status: completed
date_planned: 2026-06-25
date_completed: 2026-06-25
---

# LungCT DirectCP 4-Stage Global NCC + Dense FD-BE All-Case Sweep

## Goal

Finish the selected-case FD-BE tuning as an all-case LungCT sweep, while reusing
completed cases `01` and `08`.

## Configuration

```text
method = direct_cp_32_16_8_4_global_ncc_fd_be
cases_reused = 01,08
cases_launched = 02,03,04,05,06,07,09,10
alphas = 0,10,100,1000,10000,100000
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
lr = 1e-3
seed = 1
gpus = 0,1,2,3
```

Training loss:

```text
loss = full_crop_global_NCC + alpha * dense_forward_difference_BE
```

The training loss uses no lung-mask or border-mask weighting. The fixed T00 lung
mask is used only for foreground regularity metrics.

## Inputs And Outputs

Existing selected-case root:

```text
/database/inrorspline/runs/lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep_v1
```

New missing-case run root:

```text
/database/inrorspline/runs/lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep_v1
```

Review root:

```text
experiments/2026-06-25_lungct_directcp4stage_global_ncc_fdbe_all10_alpha_sweep/
```

## Commands

Dry run:

```bash
docker exec -w /homebase/code_sync/inrorspline py310sitreg_v1_test1 \
  python scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_global_ncc_fdbe_all10_sweep.py \
  --gpus 0,1,2,3 \
  --dry-run
```

Full launch:

```bash
docker exec -w /homebase/code_sync/inrorspline py310sitreg_v1_test1 \
  python scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_global_ncc_fdbe_all10_sweep.py \
  --gpus 0,1,2,3
```

Aggregation:

```bash
docker exec -w /homebase/code_sync/inrorspline py310sitreg_v1_test1 \
  python scripts/experiments/lungct_no_sweep_repro/aggregate_directcp4stage_global_ncc_fdbe_all10_sweep.py
```

## Reporting

Use `original_snap` TRE as the primary value. Retain all four TRE variants and
fixed-T00-lung-mask foreground regularity. Generate new all-case ellipse-style
comparison plots against IDIR BE and DirectCP 4-stage pTV LCC + isoTV.
