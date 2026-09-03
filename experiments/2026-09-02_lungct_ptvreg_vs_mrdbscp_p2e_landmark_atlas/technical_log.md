# Technical log

## Known limitations

- The first three columns are a coronal x/z projection of 3-D landmark
  vectors. The omitted y component is included in the color magnitude and in
  the scalar TRE, but is not visible as an arrow component.
- The pTVReg 4DCT MATLAB files save `Tptv_rsz`, `pts_fix`, and `pts_mov` but do
  not save `pts_moved_subvoxel`; the renderer therefore recomputes the saved
  native snap-to-voxel TRE directly from the saved field and landmarks.
- The completed COPD pTVReg reproduction uses a legacy crop origin that is one
  voxel earlier in y/x than the corrected canonical pTV-prep origin. The
  renderer aligns both methods through absolute native landmark coordinates
  and records this offset in the casewise summary.
- The registered-TRE ECDF column uses a pooled dataset-wide p99 x-limit and
  reports the number of landmarks outside that limit in each row.
- The vector plots use the canonical raw-zyx coronal display transform
  `[+x,-z]`, verified as patient R-left/L-right/S-top/I-bottom. This is a
  display transform only; the full 3-D physical-mm magnitude is unchanged.
- Each of the 30 vector panels has an independent colorbar. Initial colorbars
  use the shared ten-case initial normalization; both residual columns use the
  shared ten-case/method residual normalization.
- The atlas grid uses compact row/column spacing. Each ECDF subplot carries a
  local method-color legend: registered/applied-motion panels show pTVReg and
  MR-D-BSCP P2e, while the full TRE panels also show Initial.

## Execution record

The renderer was executed successfully in the standard `py310sitreg_v1_test1`
container on 2026-09-02. All 20 cases passed with 300 landmarks per case and
the atlas topology checks passed: 10 rows, 6 columns, 30 vector panels, and
30 colorbars per dataset. The maximum pTVReg recomputed-versus-saved
case-mean TRE difference was below `7e-16 mm`; the maximum P2e
recomputed-versus-saved landmark TRE difference was `0 mm`. The regenerated
PNG exports are 4860x6120 pixels for both datasets.

The revised notebook was then executed in the same container with
`jupyter nbconvert --to notebook --execute --inplace`. It completed without
cell errors, retained three executed code cells, and stored two inline PNG
outputs for review.
