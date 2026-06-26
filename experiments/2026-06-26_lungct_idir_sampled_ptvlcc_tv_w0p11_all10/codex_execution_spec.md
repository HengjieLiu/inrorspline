---
doc_type: codex_execution_spec
status: planned
date_planned: 2026-06-26
---

# LungCT IDIR Sampled pTV-LCC + TV 0.11 Pilot

## Objective

Run a 10-case pilot for `idir_sampled_ptv_lcc_tv` before launching any weight
sweep. This method keeps IDIR's sampled-coordinate training mechanism but uses a
pTV-style local-CC data term and a sampled analytic physical-unit TV regularizer.

## Method

- Dataset: DIR-Lab 4DCT cases `01-10`.
- Input protocol: canonical regenerated pTVReg-style landmark crop, 1mm grid.
- Training sample domain: pTV 1mm border-masked crop.
- Metric foreground: fixed T00 lung mask only.
- Loss: `sampled_ptv_lcc + 0.11 * sampled_idir_physical_tv`.
- SIREN: hidden dim `256`, hidden layers `3`, omega `32`.
- Optimization: Adam, `lr=1e-4`, `epochs=2500`, seed `1`.
- Sample settings: `512` center samples per epoch and `64` Gaussian-weighted
  integer patch offsets per center.
- pTV settings: `sigma_mm=2.1`, `csqrt=5e-3`, `border_mask_voxels=5`.

## Paths

Output root:

```text
/database/inrorspline/runs/lungct_idir_sampled_ptvlcc_tv_w0p11_all10_v1
```

Review root:

```text
experiments/2026-06-26_lungct_idir_sampled_ptvlcc_tv_w0p11_all10/
```

## Commands

Compile and loader audit:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python -m py_compile scripts/experiments/lungct_no_sweep_repro/common.py scripts/experiments/lungct_no_sweep_repro/run_one.py scripts/experiments/lungct_no_sweep_repro/launch_idir_sampled_ptvlcc_tv_w0p11_all10.py scripts/experiments/lungct_no_sweep_repro/aggregate_idir_sampled_ptvlcc_tv_pilot.py'
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/lungct_no_sweep_repro/run_one.py --audit-loader --audit-cases 1,2,3,4,5,6,7,8,9,10'
```

Case01 smoke:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python -u scripts/experiments/lungct_no_sweep_repro/run_one.py --method idir_sampled_ptv_lcc_tv --case-id 1 --gpu 0 --epochs 5 --lr 1e-4 --ptv-isotv-weight 0.11 --eval-interval 1 --out-root /database/inrorspline/runs/lungct_idir_sampled_ptvlcc_tv_w0p11_smoke_v1 --overwrite'
```

Dry-run and full launch:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/lungct_no_sweep_repro/launch_idir_sampled_ptvlcc_tv_w0p11_all10.py --gpus 0,1,2,3 --dry-run'
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python -u scripts/experiments/lungct_no_sweep_repro/launch_idir_sampled_ptvlcc_tv_w0p11_all10.py --gpus 0,1,2,3'
```

Aggregation:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/lungct_no_sweep_repro/aggregate_idir_sampled_ptvlcc_tv_pilot.py'
```

## Acceptance

- Compile passes in Docker.
- Loader audit passes for cases `01-10`.
- Smoke writes finite `loss`, `data_loss`, `reg_loss`, summary TRE, and peak
  memory.
- Dry-run queues exactly `10` jobs.
- Full launch writes `10/10` completed summaries.
- Aggregation writes `case_rows.csv`, `mean_rows.csv`, and `review_report.md`
  with no completion errors.
