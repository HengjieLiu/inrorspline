# LungCT Case08 pTVReg Fit-Modes LR Sweep Execution Spec

> Historical paths are preserved as executed. Current equivalent:
> `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.

## Objective

Repeat the Case01 fit-modes LR sweep on DIR-Lab 4DCT case `08`, using the same
methods, objectives, masks, weights, scheduling, and review plots.

## Fixed Protocol

- Dataset: 4DCT case `08`, fixed `T00`, moving `T50`.
- pTVReg source field:
  `/database/inrorspline/ptvreg/4DCT/case08_T50_to_T00_deformation.mat`.
- Data root:
  `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root:
  `/database/inrorspline/runs/lungct_case08_ptvreg_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep/`.
- Methods: IDIR, SINR cps4, DirectCP cps4, DirectCP 4-stage cps4.
- Objectives: image registration and pTVReg field-fit MSE.
- LRs: `1e-2,1e-3,1e-4`.
- Epochs: `2500`; DirectCP 4-stage uses `400,400,800,900`.
- Registration weights: IDIR `alpha_bending=10`, pTV isoTV `0.11`.
- Field-fit MSE domain: fixed T00 lung mask in the 1mm crop.

## Execution

Use method-batch scheduling so each method runs serially and its six jobs are
spread across GPUs `0,1,2,3`.

## Review Requirements

The final visualization set must match Case01:

- all LR registration loss curves
- all LR field-fit MSE curves
- final metric-vs-LR plot
- best-LR masked MSE overlay with log-scale y-axis
- best-LR IDIR registration loss plot
- best-LR spline registration loss overlay

DirectCP 4-stage curves must use a continuous global epoch axis from `1` to
`2500`.
