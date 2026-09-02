# Execution specification

## Purpose

Render an inline notebook-only visual comparison of completed COPD 1-mm pTVReg
MATLAB and MR-D-BSCP P2e registrations. No training, optimization, field write,
or data modification is allowed.

## Required environment variables

```text
PTVREG_COPD_MATLAB_ROOT
MRDBSCP_P2E_COPD_ROOT
PTVREG_ISO1MM_COPD_ROOT
```

The first points to the directory containing `case01` through `case10` MATLAB
archives. The second points to the P2e run-family directory; the third points to
the canonical `pTVReg_iso1mm/COPD` preprocessing directory. These operational
locations are intentionally supplied only through the execution environment.

## Data and field contract

- Fixed image: iBH; moving image: eBH; image-level registration direction:
  `eBH -> iBH`.
- The canonical P2e output grid is the shared display grid.
- pTVReg `Tptv_rsz` is native full-grid, physical-mm, component order `x,y,z`,
  and is a fixed-grid pull field used to sample the native moving image.
- P2e `disp_vox_reg1mm_zyx.npy` is canonical-grid, 1-mm voxel, component order
  `z,y,x`. It is converted to normalized `x,y,z` only for pull sampling.
- CT sampling is trilinear with `align_corners=True`; lung-mask sampling is
  nearest-neighbor with the same coordinates.

## Render protocol

- Cases: COPD 01–10 / challenge 104–113.
- Coronal AP locations: fixed-lung occupied-y-range fractions
  `0.15, 0.35, 0.50, 0.65, 0.85`.
- Eight columns per row: moving, fixed, pTVReg deformed, P2e deformed, pTVReg
  residual, P2e residual, pTVReg magnitude, P2e magnitude.
- All rows and methods within a case share full-grid 99th-percentile limits:
  symmetric for residual, nonnegative for magnitude.
- The output display uses the established raw COPD transform and explicitly
  marks S/I/R/L. AP remains the reported slice-normal coordinate.

## Required checks

1. Validate all input arrays and common-grid shapes.
2. Validate pTVReg component/unit/direction at saved landmarks and compare the
   reconstructed native deformed volume with the saved MATLAB diagnostic.
3. Validate P2e voxel field against saved normalized coordinates and its
   per-landmark subvoxel TRE table.
4. Confirm every selected slice intersects the fixed lung and all output arrays
   are finite.
5. Execute the notebook inside `py310sitreg_v1_test1` and retain all ten inline
   case figures.
