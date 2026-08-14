# LungCT 2500-Step Schedule Standardization Pilot - Codex Execution Spec

## Objective

Run two all-10 DIR-Lab/LungCT pilot reruns with the standardized multistage
schedule `400,400,800,900` epochs and report whether the new 2500-step budget
matches or improves the existing 2400-step operating points.

## Fixed Schedule

- Cases: `01-10`.
- Seed: `1`.
- GPUs: `0,1,2,3`.
- Stage full control points: `32,16,8,4`.
- Stage downsample: `8,4,2,1`.
- Stage epochs: `400,400,800,900`.
- Total epochs: `2500`.

## Pilot A: pTVReg-Loss MR-D-BSCP

- Method: `direct_cp_32_16_8_4_ptv_lcc_isotv`.
- Loss: sampled pTVReg-style LCC plus isotropic TV.
- Weight: `ptv_isotv_weight=0.11`.
- LR: `1e-3`.
- Output root:
  `${DATA_ROOT}/runs/lungct_directcp32_ptvlcc_w0p11_e400_400_800_900_all10_v1`.
- Review folder:
  `experiments/2026-08-12_lungct_directcp32_ptvlcc_w0p11_schedule_standardization/`.
- Baseline comparison:
  `experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/summary_runs.csv`.

## Pilot B: Corrected IDIR-Loss MR-D-BSCP

- Method: `direct_cp_32_16_8_4_masked_ncc_bspline_be`.
- Loss: fixed T00 lung-mask sampled NCC plus analytic cubic B-spline BE.
- Alpha: `33333.333333333336`.
- LR: `1e-3`.
- Output root:
  `${DATA_ROOT}/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha33333_e400_400_800_900_all10_v1`.
- Review folder:
  `experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha33333_e400_400_800_900_all10/`.
- Baseline comparison:
  `experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha_sweep/mean_rows.csv`.

## Commands

Dry-run Pilot A:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_lungct_2500_schedule_pilots.py --pilots ptvreg_w0p11 --gpus 0,1,2,3 --dry-run
```

Launch Pilot A:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_lungct_2500_schedule_pilots.py --pilots ptvreg_w0p11 --gpus 0,1,2,3
```

Dry-run Pilot B:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_lungct_2500_schedule_pilots.py --pilots masked_alpha33333 --gpus 0,1,2,3 --dry-run
```

Launch Pilot B:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_lungct_2500_schedule_pilots.py --pilots masked_alpha33333 --gpus 0,1,2,3
```

Aggregate:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/aggregate_lungct_2500_schedule_pilots.py
```

## Acceptance Checks

- Exactly 10 completed summaries for each pilot.
- Every summary records `training.stage_epochs == [400, 400, 800, 900]`.
- No summary path contains the old `e200-400-800-1000` schedule token.
- pTVReg-loss pilot keeps preprocessing, loss, LR, seed, and `w=0.11` fixed.
- Corrected IDIR-loss pilot uses fixed T00 lung-mask sampled NCC and analytic
  cubic B-spline BE.
- Reports include aggregate and case-by-case TRE, foreground stdLogJ,
  `J<=0%`, elapsed time, and peak memory.
