---
doc_type: technical_log
status: completed
date_started: 2026-07-07
date_completed: 2026-07-07
---

# Technical Log

## Known Limitations

- This run changes only the stage epoch allocation from P2d
  `100,200,400,800,1000` to `400,400,400,400,900`.
- The method identifier remains `direct_cp_32_16_8_4_ptv_lcc_isotv` for code
  compatibility; the run name and training metadata record the actual
  `64,32,16,8,4` stage list and `400,400,400,400,900` epoch schedule.
- COPD preprocessing is reused from the completed pTV-prep root.
- Headline TRE is native-crop/original-domain snap-to-voxel TRE; reg-1mm TRE is
  diagnostic.
- Because the code path and stage list are unchanged from P2d, the completed
  P2d case01 `1,1,1,1,1` training smoke is reused; this run still performs
  compile and launcher dry-run checks before the full launch.

## Verification Checklist

- [x] Compile changed scripts with `python -m py_compile`.
- [x] Confirm GPUs `0,1,2,3` are available.
- [x] Dry-run launcher with `400,400,400,400,900`; require 10 queued jobs and
  schedule-specific run names.
- [x] Launch cases `01-10` on GPUs `0,1,2,3`; require `10/10` completed summaries.
- [x] Aggregate `copd_case_rows.csv`, `copd_mean_row.csv`, and `review_report.md`.

## Results

- Full run root:
  `/database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity_v1`.
- Review report:
  `experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity/review_report.md`.
- Completed summaries: `10/10`.
- Headline native-crop/original snap TRE mean:
  `1.0505 +/- 0.7673 mm`.
- Per-case headline native-crop/original snap TRE:
  `0.714, 3.348, 0.823, 0.732, 0.740, 0.851, 0.836, 0.820, 0.851, 0.789`.
- Diagnostic native subvoxel TRE mean:
  `1.1567 +/- 0.7370 mm`.
- Diagnostic reg-1mm snap TRE mean:
  `1.2914 +/- 0.7297 mm`.
- Mean foreground SDlogJ:
  `0.4021`.
- Mean foreground `J<=0` percentage:
  `0.7251`.
- Dense displacement arrays were saved for all 10 cases as
  `disp_norm_xyz_reg1mm.npy` and `disp_vox_reg1mm_zyx.npy`.
- Compared with P2d (`100,200,400,800,1000`), the mean headline TRE improved
  from `2.5026` to `1.0505 mm`; the largest case improvements were cases `04`,
  `05`, and `08`, while case `02` remains the main outlier.
