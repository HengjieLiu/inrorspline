# LungCT pTVReg vs MR-D-BSCP P2e landmark residual atlas

This is a posthoc visualization of the completed pTVReg reproduction and
MR-D-BSCP P2e fields for all ten 4DCT and all ten COPD cases. It does not run
registration.

The selected P2e variants are fixed by the best aggregate native-crop snap
TRE from `2026-09-01_lungct_mrdbscp_acceleration_best1mm`:

- 4DCT: `original_corrected`
- COPD: `oasis_bal_decay_ordinal`

Run the notebook in the standard container after setting these variables to
the corresponding local data locations:

```bash
export INRORSPLINE_DIRLAB_PTVREG_ISO1MM_ROOT=/path/to/processed/pTVReg_iso1mm
export PTVREG_4DCT_ROOT=/path/to/pTVReg/4DCT
export PTVREG_COPD_ROOT=/path/to/pTVReg/COPD/reproduction/gpu
export MRDBSCP_P2E_ACCEL_ROOT=/path/to/mrdbscp/p2e/acceleration/direct_cp_32_16_8_4_ptv_lcc_isotv
```

The headline TRE values use native-crop snap-to-voxel evaluation for both
methods. Vector panels use local physical-mm positions, with arrows shown in
the verified coronal display orientation: patient R left, L right, S top, and
I bottom. The raw-zyx z rows are flipped for display; color still encodes the
full 3-D magnitude. Every vector panel has its own colorbar, while the initial
bars share one global ten-case scale and the residual bars share one global
ten-case/method scale. Applied-motion ECDFs are evaluated on the fixed lung
mask in the canonical 1-mm grid.

The executed notebook writes the two requested PNGs plus a casewise CSV and a
provenance JSON under `reports/`.
