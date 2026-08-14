# LungCT Figure 2 MR-D-BSCP 2500-Step Schedule Refresh - Codex Execution Spec

## Objective

Rerun Figure 2 MR-D-BSCP curve points using `400,400,800,900` stage epochs
while preserving the existing plotted method/loss definitions and all other
settings.

## Fixed Settings

- Cases: DIR-Lab/LungCT `01-10`.
- Seed: `1`.
- LR: `1e-3`.
- GPUs: `0,1,2,3`.
- Stage full control points: `32,16,8,4`.
- Stage downsample: `8,4,2,1`.
- Stage epochs: `400,400,800,900`.
- Data root: `${LUNGCT_PTVREG_ISO1MM}`.

## Run Families

- MR-D-BSCP (IDIR loss): `direct_cp_32_16_8_4_global_ncc_fd_be`.
  - Loss remains full-crop global NCC plus dense forward-difference BE.
  - Alphas: `1000,3000,10000,30000,100000`.
  - Output root:
    `${DATA_ROOT}/runs/lungct_fig2_mrdbscp_idirloss_fdbe_e400_400_800_900_v1`.

- MR-D-BSCP (pTVReg loss): `direct_cp_32_16_8_4_ptv_lcc_isotv`.
  - Loss remains pTV-LCC plus isoTV.
  - Weights: `0.011,0.033,0.055,0.11,0.33`.
  - Reuse existing completed 2500-step `w=0.11` anchor from:
    `${DATA_ROOT}/runs/lungct_directcp32_ptvlcc_w0p11_e400_400_800_900_all10_v1`.
  - Output root for new weights:
    `${DATA_ROOT}/runs/lungct_fig2_mrdbscp_ptvloss_e400_400_800_900_v1`.

## Commands

Dry-run IDIR-loss:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_lungct_fig2_mrdbscp_2500_schedule_refresh.py --families idir_fdbe --gpus 0,1,2,3 --dry-run
```

Launch IDIR-loss:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_lungct_fig2_mrdbscp_2500_schedule_refresh.py --families idir_fdbe --gpus 0,1,2,3
```

Dry-run pTVReg-loss:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_lungct_fig2_mrdbscp_2500_schedule_refresh.py --families ptv --gpus 0,1,2,3 --dry-run
```

Launch pTVReg-loss:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_lungct_fig2_mrdbscp_2500_schedule_refresh.py --families ptv --gpus 0,1,2,3
```

Aggregate:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/aggregate_lungct_fig2_mrdbscp_2500_schedule_refresh.py
```

## Acceptance

- IDIR-loss dry-run queues exactly `50` jobs.
- pTVReg-loss dry-run queues exactly `40` jobs and reuses `10` completed
  `w=0.11` anchor jobs.
- Every new summary records `training.stage_epochs == [400,400,800,900]`.
- New IDIR-loss rows use `direct_cp_32_16_8_4_global_ncc_fd_be`.
- New pTVReg-loss rows use `direct_cp_32_16_8_4_ptv_lcc_isotv`.
- No new summary path contains `e200-400-800-1000`.
- Reports compare old and new schedules for every requested alpha/weight.
