---
doc_type: technical_log
status: full_complete
experiment_id: lungct_dual_inr_1mm_lungmask_author_schedule_v1
date_planned: 2026-08-07
---

# Technical Log

## Known Limitations And Warnings

- This is a protocol-tagged comparison, not a clean method-only comparison
  against older native IDIR, pTVReg, DirectCP/SINR, or author DUAL rows.
- `PL1A` keeps the `PL` pTVReg 1 mm fixed T00 lung-mask domain but uses the
  released DUAL blur schedule. It is not the author A1 row, because A1 used the
  author five-lobe crop/sampling setup.
- Original-schedule transitions are adaptive. Per-case stage lengths are
  realized outcomes and must be read from `stage_trace.csv/json`.

## Run Ledger

| Stage | Status | Evidence |
|---|---|---|
| implementation | complete | `PL1A` variant, author-blur training path, and aggregation contract added |
| unit tests | passed | Docker focused tests `15 passed in 1.34s` |
| full cohort | passed | cases01-10 for `PL1A`, 10/10 accepted and 10/10 regularity complete |

## 2026-08-07 Full-Cohort Notes

- Full launch used GPUs `0,1,2,3`, cases `01-10`, variant `PL1A`, seed `42`.
- All `10/10` records completed on attempt 1.
- Final aggregation passed with `10/10` accepted optimization records and
  `10/10` completed common posthoc regularity records.
- Native snap TRE was `0.9758 +/- 0.1752 mm`; reg-grid subvoxel TRE was
  `1.1144 +/- 0.1280 mm`.
- Foreground SDlogJ was `0.143408 +/- 0.037411`; foreground `J<=0` percentage
  was `0.0 +/- 0.0`.
- PL1A beat pTVReg native snap TRE in `3/10` cases and MR-D-BSCP pTV-loss
  `w=0.055` in `3/10` cases. It beat PL1D in `5/10` cases.
- Detailed full-cohort comparison:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_author_schedule/full_cohort_comparison.md`.
