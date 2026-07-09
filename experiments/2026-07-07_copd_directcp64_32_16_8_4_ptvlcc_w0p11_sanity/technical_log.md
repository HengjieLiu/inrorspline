---
doc_type: technical_log
status: completed
date_started: 2026-07-07
date_completed: 2026-07-07
---

# Technical Log

## Known Limitations

- This run changes only the multistage pyramid from `32,16,8,4` to
  `64,32,16,8,4`, with total epochs fixed at `2500`.
- The method identifier remains `direct_cp_32_16_8_4_ptv_lcc_isotv` for code
  compatibility; the run name and training metadata record the actual
  `64,32,16,8,4` stage list.
- COPD preprocessing is reused from the completed pTV-prep root. Case `01`
  uses the plan-compatible `[275,201,229]` pTV 1mm grid.
- Headline TRE is native-crop/original-domain snap-to-voxel TRE; reg-1mm TRE is
  diagnostic.

## Verification Checklist

- [x] Compile changed scripts with `python -m py_compile`.
- [x] Run case `01` smoke with stage epochs `1,1,1,1,1`; require completed summary,
  finite headline TRE, finite SDlogJ, and saved dense DVF.
- [x] Launch cases `01-10` on GPUs `0,1,2,3`; require `10/10` completed summaries.
- [x] Aggregate `copd_case_rows.csv`, `copd_mean_row.csv`, and `review_report.md`.

## Results

- Full run root:
  `/database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity_v1`.
- Review report:
  `experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity/review_report.md`.
- Completed summaries: `10/10`.
- Headline native-crop/original snap TRE mean:
  `2.5026 +/- 2.2543 mm`.
- Per-case headline native-crop/original snap TRE:
  `1.303, 3.335, 0.823, 7.261, 5.873, 0.986, 0.849, 3.164, 0.639, 0.793`.
- Diagnostic native subvoxel TRE mean:
  `2.5933 +/- 2.2404 mm`.
- Diagnostic reg-1mm snap TRE mean:
  `2.7184 +/- 2.2283 mm`.
- Mean foreground SDlogJ:
  `0.4778`.
- Mean foreground `J<=0` percentage:
  `2.1468`.
- Dense displacement arrays were saved for all 10 cases as
  `disp_norm_xyz_reg1mm.npy` and `disp_vox_reg1mm_zyx.npy`.
- Motion-range follow-up table:
  `copd_motion_range_vs_post_tre.csv` and
  `copd_motion_range_vs_post_tre.md`. Across all 3000 COPD landmarks,
  pre-registration TRE is mean `23.3606 mm`, P90 `39.0344 mm`,
  P95 `44.9894 mm`, and max `66.4997 mm`; post snap TRE mean is
  `2.5026 mm`.

Compared with P2c (`32,16,8,4`, `200,400,800,1000`), adding the `64` stage and
redistributing the same 2500 epochs substantially improved the COPD headline
mean TRE from `6.8486` to `2.5026 mm`, most visibly rescuing cases `01`, `04`,
`05`, `06`, and `10`.
