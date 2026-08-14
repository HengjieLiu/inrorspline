# LungCT 2500-Step Schedule Standardization Pilot - Technical Log

## Known Limitations

- This is a pilot check, not a full hyperparameter sweep.
- The pTVReg-loss comparison should be interpreted as a schedule-only check
  against the existing `w=0.11` 2400-step all-10 result.
- The corrected IDIR-loss comparison changes both schedule and alpha relative
  to the completed `alpha=10000` and `alpha=100000` 2400-step anchors because
  the requested pilot point is `alpha=1e5/3`.
- All LungCT rows still require protocol tags before method-only claims across
  pTVReg-loss, IDIR-loss, DUAL-INR, and PTVReg families.

## 2026-08-12 Setup

- Added launcher:
  `scripts/experiments/lungct_no_sweep_repro/launch_lungct_2500_schedule_pilots.py`.
- Added aggregator:
  `scripts/experiments/lungct_no_sweep_repro/aggregate_lungct_2500_schedule_pilots.py`.
- Registered run as P2m in `docs/04_experiment_registry.md`.
- Planned Docker container:
  `py310sitreg_v1_test1`.
- Planned output roots:
  `${DATA_ROOT}/runs/lungct_directcp32_ptvlcc_w0p11_e400_400_800_900_all10_v1`
  and
  `${DATA_ROOT}/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha33333_e400_400_800_900_all10_v1`.

## 2026-08-12 Dry-Run Validation

- No-bytecode syntax check passed for launcher and aggregator.
- Pilot A dry-run queued exactly `10` jobs, cases `01-10`, GPUs `0,1,2,3`,
  schedule `400,400,800,900`, and run names containing
  `e400-400-800-900`.
- Pilot B dry-run queued exactly `10` jobs, cases `01-10`, GPUs `0,1,2,3`,
  schedule `400,400,800,900`, run names containing `e400-400-800-900`,
  and `alpha=33333.333333333336`.

## 2026-08-12 Pilot A Completion

- Pilot A completed `10/10` cases with return code `0`.
- Aggregation with `--pilots ptvreg_w0p11` passed with errors `[]`.
- 2400-step pTVReg-loss baseline `w=0.11`: original snap TRE
  `0.922 +/- 0.152 mm`, foreground stdLogJ
  `0.179961 +/- 0.049568`, foreground `J<=0%`
  `0.0138 +/- 0.0193`.
- 2500-step pTVReg-loss pilot `w=0.11`: original snap TRE
  `0.924 +/- 0.152 mm`, foreground stdLogJ
  `0.179973 +/- 0.049585`, foreground `J<=0%`
  `0.0137 +/- 0.0191`.
- Interpretation so far: the standardized 2500-step schedule does not
  materially change the best pTVReg-loss point.

## 2026-08-12 Pilot B Completion And Final Aggregation

- Pilot B completed `10/10` cases with return code `0`.
- Final aggregation over both pilots passed with errors `[]`.
- 2400-step corrected IDIR-loss `alpha=10000`: original snap TRE
  `1.037 +/- 0.153 mm`, foreground stdLogJ
  `0.265395 +/- 0.063547`, foreground `J<=0%`
  `0.0349 +/- 0.0579`.
- 2400-step corrected IDIR-loss `alpha=100000`: original snap TRE
  `1.003 +/- 0.154 mm`, foreground stdLogJ
  `0.155840 +/- 0.033335`, foreground `J<=0%`
  `0.0005 +/- 0.0014`.
- 2500-step corrected IDIR-loss `alpha=33333.333333`: original snap TRE
  `1.016 +/- 0.160 mm`, foreground stdLogJ
  `0.195866 +/- 0.042067`, foreground `J<=0%`
  `0.0058 +/- 0.0151`.
- Interpretation: `alpha=33333.333333` is a reasonable intermediate point but
  does not beat the existing 2400-step `alpha=100000` corrected-IDIR result.
- Stop condition satisfied: both requested pilots are complete and no full
  2500-step sweeps were launched.
