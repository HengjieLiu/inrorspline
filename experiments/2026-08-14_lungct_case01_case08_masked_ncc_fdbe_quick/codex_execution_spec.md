# Codex execution spec: LungCT masked-NCC FD-BE quick A/B

## Objective

Run a controlled two-case check comparing prior unmasked full-crop NCC FD-BE
training against new fixed-T00 lung-mask NCC FD-BE training for INR-BSCP,
D-BSCP, and MR-D-BSCP.

## Code Changes Required

- Add these method IDs to `scripts/experiments/lungct_no_sweep_repro/run_one.py`:
  - `sinr_cps4_masked_ncc_fd_be`
  - `direct_cp_cps4_masked_ncc_fd_be`
  - `direct_cp_32_16_8_4_masked_ncc_fd_be`
- For these methods, use fixed T00 lung-mask voxels for global NCC data loss.
- Keep dense forward-difference BE unmasked and use alpha `10000`.
- Preserve existing full-crop/lung-mask NCC diagnostics.

## Commands

Compile:

```bash
docker exec py310sitreg_v1_test1 python -m py_compile \
  /workspace/repo/scripts/experiments/lungct_no_sweep_repro/run_one.py \
  /workspace/repo/scripts/experiments/lungct_no_sweep_repro/launch_lungct_case01_case08_masked_ncc_fdbe_quick.py \
  /workspace/repo/scripts/experiments/lungct_no_sweep_repro/aggregate_lungct_case01_case08_masked_ncc_fdbe_quick.py
```

Smoke tests:

```bash
docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method sinr_cps4_masked_ncc_fd_be --case-id 1 \
  --data-root /database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm \
  --out-root /database/inrorspline/runs/lungct_case01_case08_masked_ncc_fdbe_quick_smoke_v1 \
  --gpu 1 --seed 1 --epochs 1 --lr 1e-4 --alpha 10000 \
  --eval-interval 1 --global-ncc-diagnostic-interval 1 --no-save-dense-dvf --overwrite

docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method direct_cp_cps4_masked_ncc_fd_be --case-id 1 \
  --data-root /database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm \
  --out-root /database/inrorspline/runs/lungct_case01_case08_masked_ncc_fdbe_quick_smoke_v1 \
  --gpu 2 --seed 1 --epochs 1 --lr 1e-3 --alpha 10000 \
  --eval-interval 1 --global-ncc-diagnostic-interval 1 --no-save-dense-dvf --overwrite

docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method direct_cp_32_16_8_4_masked_ncc_fd_be --case-id 1 \
  --data-root /database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm \
  --out-root /database/inrorspline/runs/lungct_case01_case08_masked_ncc_fdbe_quick_smoke_v1 \
  --gpu 3 --seed 1 --lr 1e-3 --alpha 10000 \
  --stage-full-cps 32,16,8,4 --stage-downsample 8,4,2,1 --stage-epochs 1,1,1,1 \
  --eval-interval 1 --global-ncc-diagnostic-interval 1 --no-save-dense-dvf --overwrite
```

Dry-run and full launch:

```bash
docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/launch_lungct_case01_case08_masked_ncc_fdbe_quick.py \
  --dry-run --cases 1,8 --gpus 1,2,3

docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/launch_lungct_case01_case08_masked_ncc_fdbe_quick.py \
  --cases 1,8 --gpus 1,2,3 --resume
```

Aggregate:

```bash
docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/aggregate_lungct_case01_case08_masked_ncc_fdbe_quick.py
```

## Acceptance

- All six full jobs complete.
- Every new summary has `loss_domain=fixed_t00_lung_mask_global_ncc`,
  `mask_rule=fixed_t00_lung_mask_loss_and_metric`,
  `training.image_loss=fixed_t00_lung_mask_global_ncc`, and
  `training.regularizer=dense_forward_difference_be`.
- The aggregate report includes prior unmasked and new masked TRE, stdLogJ,
  folding percentage, full-crop 1-NCC, fixed-mask 1-NCC, runtime, peak memory,
  final data loss, and final regularizer loss.
