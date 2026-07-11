---
doc_type: codex_execution_spec
status: completed
date_started: 2026-07-07
date_completed: 2026-07-07
---

# COPD DirectCP64-32-16-8-4 pTV-LCC + isoTV E400x4+900 Sanity

## Purpose

Rerun the COPD `64,32,16,8,4` pTVReg-loss DirectCP sanity experiment with the
same total `2500` epochs as P2d, but redistribute epochs as
`400,400,400,400,900`.

## Protocol

- Dataset: COPD cases `01-10`.
- Direction: `eBHCT -> iBHCT`.
- Protocol tag: `ptv_landmark_crop_1mm_no_lungmask`.
- Preprocessing root:
  `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Method: `direct_cp_32_16_8_4_ptv_lcc_isotv` runner with custom stage lists.
- Stage full cps: `64,32,16,8,4`.
- Stage downsample: `16,8,4,2,1`.
- Stage epochs: `400,400,400,400,900`.
- Total epochs: `2500`.
- Learning rate: `1e-3`.
- Seed: `1`.
- pTV LCC sigma: `2.1 mm`.
- isoTV weight: `0.11`.
- `csqrt`: `5e-3`.
- Border mask: `5`.
- Save dense displacement arrays: yes.

## Output

- Run root:
  `/database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity_v1`.
- Log root:
  `/homebase/code_sync/inrorspline/logs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity_v1`.
- Review root:
  `experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity`.

## Verification

- Compile runner, launcher, and aggregator with `python -m py_compile`.
- Reuse the completed P2d `64,32,16,8,4` case01 `1,1,1,1,1` smoke for code
  path validation; this run changes only the epoch allocation.
- Dry-run launcher to confirm 10 queued case commands and schedule-specific run
  names.
- Launch all 10 cases on GPUs `0,1,2,3`.
- Aggregate `copd_case_rows.csv`, `copd_mean_row.csv`, and `review_report.md`.
