# Codex execution specification

## Scope

Posthoc rendering only. No registration, preprocessing, dataset movement, or
checkpoint modification is performed.

## Inputs

- Canonical pTVReg-style 1-mm LungCT case directories for 4DCT and COPD.
- Ten completed pTVReg MATLAB deformation files for each dataset.
- Ten completed MR-D-BSCP P2e dense fields and native-crop landmark CSVs for
  the selected variant of each dataset.

## Required validation

- Every case has 300 finite fixed and moving landmarks.
- pTVReg recomputed native snap TRE mean and sample standard deviation match
  saved `TRE_phys` and `TREstd_case`.
- P2e recomputed native snap TRE matches `snap_tre_mm` in the saved CSV.
- pTVReg and P2e initial landmark vectors agree in the canonical crop.
- Coronal coordinate-orientation self-check passes: patient R is left, patient
  L is right, patient S is top, and patient I is bottom.
- All 60 main axes retain x/y tick labels and axis labels; all 30 vector axes
  have independent colorbars bound to the shared initial/residual scales.

## Outputs

- `reports/4dct_ptvreg_vs_mrdbscp_p2e_landmark_atlas.png`
- `reports/copd_ptvreg_vs_mrdbscp_p2e_landmark_atlas.png`
- `reports/casewise_landmark_atlas_summary.csv`
- `reports/provenance.json`

Execution is performed from this directory inside
`py310sitreg_v1_test1` using the environment variables in `README.md`.
