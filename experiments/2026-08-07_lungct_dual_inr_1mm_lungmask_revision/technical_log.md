---
doc_type: technical_log
status: full_complete
experiment_id: lungct_dual_inr_1mm_lungmask_revision_v1
date_planned: 2026-08-07
---

# Technical Log

## Known Limitations And Warnings

- This branch changes the pTV 1 mm optimization domain by using the fixed T00
  lung mask. It must not be merged into or reported as the existing `P`
  border-mask protocol.
- The `dual_native_reg` and `idir_type_be` loss branches have different
  regularization definitions and raw weights are not directly comparable.
- The IDIR-type BE branch preserves the historical coordinate-autograd
  batch-size scaling convention. Its default pilot weight is `10.0`, matching
  the repo-local LungCT IDIR default rather than a tuned DUAL-specific value.
- The pilot cases `01` and `08` are technical/protocol checks only. Do not use
  them as a performance-selected full-cohort result.
- Technical completion is based on artifacts, finite metrics, protocol
  identity, stage traces, and posthoc regularity. TRE and regularity magnitude
  do not accept or reject a completed record.

## Implementation Notes

- Added protocol `PL` in
  `scripts/experiments/lungct_dual_inr_controlled/data.py`.
- Added variants `PL1D` and `PL1B` in
  `scripts/experiments/lungct_dual_inr_controlled/adapter.py`.
- Added isolated launcher
  `scripts/experiments/lungct_dual_inr_controlled/launch_lungmask_revision.py`.
- Added isolated aggregator
  `scripts/experiments/lungct_dual_inr_controlled/aggregate_lungmask_revision.py`.

## Run Ledger

| Stage | Status | Evidence |
|---|---|---|
| implementation | complete | protocol/variant/launcher/aggregator code added |
| unit tests | passed | Docker focused tests `15 passed in 4.60s` |
| dry-run manifest | passed | four planned pilot jobs, `PL1D/PL1B` cases `01/08` |
| smoke | passed | one-step `PL1D` and `PL1B` case01 completed on attempt 1 |
| pilot | passed | full-length cases01/08 completed and aggregated, 4/4 accepted |
| full cohort | passed | cases01-10 for `PL1D` and `PL1B`, 20/20 accepted and 20/20 regularity complete |

## 2026-08-07 Pilot Notes

- Smoke launched in the default project Docker container with visible GPUs
  `0,1,2,3`. `PL1D case01` completed on attempt 1 in `62.7 s`; `PL1B
  case01` completed on attempt 1 in `87.3 s`.
- Full pilot launched immediately after smoke for cases `01` and `08`, variants
  `PL1D` and `PL1B`, seed `42`, output root
  `${RUN_ROOT}`.
- All four full-length pilot records completed on attempt 1. Wall times were:
  `PL1D case01` `378.8 s`, `PL1D case08` `417.0 s`, `PL1B case08`
  `591.9 s`, and `PL1B case01` `614.5 s`.
- Initial aggregation exposed a review-script bug: zero-byte `DONE` marker
  files were incorrectly treated as invalid. No training outputs were rerun.
  The aggregator was fixed to allow empty marker files and rerun on the
  completed records.
- Final aggregation passed with `4/4` accepted optimization records and `4/4`
  completed common posthoc regularity records.
- Pilot mean native snap TRE was `0.9282 +/- 0.2310 mm` for `PL1D` and
  `0.9481 +/- 0.1793 mm` for `PL1B`. Foreground SDlogJ was
  `0.125992 +/- 0.041732` for `PL1D` and `0.149517 +/- 0.048923` for `PL1B`.
  Both branches had foreground `J<=0` percentage `0.0`.

## 2026-08-07 Full-Cohort Launch Notes

- User approved scale-up after pilot review.
- Planned matrix: `PL1D` and `PL1B` on DIR-Lab cases `01-10`, protocol `PL`,
  seed `42`, stage lengths `400,400,800,900`, image factors `8,4,2,1`.
- Existing completed pilot records for cases `01` and `08` may be reused by the
  launcher if their `DONE` markers are present. Remaining records are submitted
  across all available GPUs.
- Full launch used GPUs `0,1,2,3`, reused `PL1D/PL1B` cases `01` and `08`, and
  completed all other records on attempt 1.
- Final aggregation passed with `20/20` accepted optimization records and
  `20/20` completed common posthoc regularity records.
- Full-cohort native snap TRE was `0.9815 +/- 0.1686 mm` for `PL1D` and
  `0.9865 +/- 0.1689 mm` for `PL1B`. Reg-grid subvoxel TRE was
  `1.1174 +/- 0.1289 mm` for `PL1D` and `1.1068 +/- 0.1296 mm` for `PL1B`.
- Compared with pTVReg native snap TRE `0.9191 +/- 0.1607 mm` and MR-D-BSCP
  pTV-loss `w=0.055` native snap TRE `0.9175 +/- 0.1561 mm`, PL1D won `3/10`
  cases against each reference and PL1B won `0/10` against pTVReg and `2/10`
  against MR-D-BSCP.
- Detailed full-cohort comparison:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/full_cohort_comparison.md`.
