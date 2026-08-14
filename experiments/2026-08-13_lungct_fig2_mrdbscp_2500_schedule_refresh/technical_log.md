# LungCT Figure 2 MR-D-BSCP 2500-Step Schedule Refresh - Technical Log

## Known Limitations

- This refresh changes schedule only. It does not alter preprocessing, loss,
  regularizer implementation, LR, seed, or metric conventions.
- The IDIR-loss family intentionally uses the current plotted
  `direct_cp_32_16_8_4_global_ncc_fd_be` protocol. It does not use the newer
  corrected fixed-mask NCC + analytic B-spline BE protocol.
- The report is review-only and must not update Figure 2 plotting sources until
  the comparison table is inspected.

## 2026-08-13 Setup

- Added launcher:
  `scripts/experiments/lungct_no_sweep_repro/launch_lungct_fig2_mrdbscp_2500_schedule_refresh.py`.
- Added aggregator:
  `scripts/experiments/lungct_no_sweep_repro/aggregate_lungct_fig2_mrdbscp_2500_schedule_refresh.py`.
- Registered run as P2n in `docs/04_experiment_registry.md`.
- Planned IDIR-loss alphas: `1000,3000,10000,30000,100000`.
- Planned pTVReg-loss weights: `0.011,0.033,0.055,0.11,0.33`.
- Existing 2500-step `w=0.11` pTVReg-loss anchor will be reused if validation
  passes.

## 2026-08-13 Dry-Run Validation

- No-bytecode syntax check passed for launcher and aggregator.
- IDIR-loss dry-run queued exactly `50/50` jobs across cases `01-10` and
  alphas `1000,3000,10000,30000,100000`.
- pTVReg-loss dry-run requested `50` jobs, queued exactly `40`, and reused
  `10` completed `w=0.11` anchor jobs from the prior 2500-step pilot.
- Both dry-runs used schedule `400,400,800,900`, stage control points
  `32,16,8,4`, downsample `8,4,2,1`, and GPUs `0,1,2,3`.

## 2026-08-13 Completion

- IDIR-loss refresh completed all `50/50` launched jobs with return code `0`.
- pTVReg-loss refresh completed all `40/40` launched jobs with return code `0`
  and reused the validated `10/10` `w=0.11` anchor summaries.
- Aggregation completed with no validation errors and wrote:
  - `case_rows.csv/json`
  - `mean_rows.csv/json`
  - `schedule_comparison_rows.csv/json`
  - `manifest.json`
  - `review_report.md`
- Aggregate row counts: `200` case rows, `20` mean rows, and `10`
  old-vs-new comparison rows.
- No Figure 2 plotting source was updated.
