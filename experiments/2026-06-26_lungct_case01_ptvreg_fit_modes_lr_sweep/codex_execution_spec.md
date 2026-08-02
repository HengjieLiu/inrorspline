# LungCT Case01 pTVReg Fit-Modes LR Sweep Execution Spec

> Historical paths are preserved as executed. Current equivalent:
> `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.

## Objective

Run a single-case diagnostic comparing IDIR, SINR cps4, DirectCP cps4, and
DirectCP 4-stage under two objectives:

- image registration on canonical pTVReg-style case01 1mm crop data
- direct MSE fitting to the saved pTVReg case01 displacement field

## Fixed Protocol

- Case: 4DCT case `01`, fixed `T00`, moving `T50`.
- Data root:
  `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- pTVReg field:
  `/database/inrorspline/ptvreg/4DCT/case01_T50_to_T00_deformation.mat`.
- Learning rates: `1e-2`, `1e-3`, `1e-4`.
- Schedule: run one method at a time, distributing that method's six jobs
  across GPUs `0,1,2,3`.
- DirectCP 4-stage: `32,16,8,4`; downsample `8,4,2,1`; epochs
  `400,400,800,900`.

## Loss And Mask Rules

- IDIR registration: fixed T00 lung-mask sampled NCC plus IDIR BE,
  `alpha_bending=10`, coordinate batch size `10000`.
- SINR/DirectCP registration: pTV LCC plus isoTV with `ptv_isotv_weight=0.11`,
  `sigma=2.1mm`, `csqrt=5e-3`, and 5-voxel border mask.
- Field fitting: MSE only over the fixed T00 lung mask in the 1mm crop.
- Full-crop MSE is retained as a diagnostic and is not optimized.
- Foreground regularity metrics use the fixed T00 lung mask.

## pTVReg Field Validation

The `.mat` file is MATLAB v7.3/HDF5. The loader must validate:

- keys `Tptv_rsz`, `pts_fix`, `pts_mov`, `spc_orig_saved`, `crop_v_saved`,
  `TRE_phys`, and `TREstd_case`
- `Tptv_rsz` h5py shape `(3,94,256,256)`, interpreted as MATLAB
  `T(x,y,z,component)`
- physical-mm components in `x,y,z`, divided by saved spacing before moving
  fixed landmarks
- exact pTVReg snap-to-voxel TRE: `0.7671410824`

## Launch Commands

Dry run:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/lungct_no_sweep_repro/launch_case01_ptvreg_fit_modes_lr_sweep.py --dry-run --method-gpu-map idir:0,sinr_cps4:1,direct_cp_cps4:2,direct_cp_4stage:3'
```

Full run:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/lungct_no_sweep_repro/launch_case01_ptvreg_fit_modes_lr_sweep.py --method-gpu-map idir:0,sinr_cps4:1,direct_cp_cps4:2,direct_cp_4stage:3 --epochs 2500 --stage-epochs 400,400,800,900 --eval-interval 250'
```

Restarted full launch uses method-batch scheduling and overwrites partial
outputs from the stopped method-pinned launch:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/lungct_no_sweep_repro/launch_case01_ptvreg_fit_modes_lr_sweep.py --schedule-mode method_batches --gpus 0,1,2,3 --epochs 2500 --stage-epochs 400,400,800,900 --eval-interval 250 --overwrite'
```
