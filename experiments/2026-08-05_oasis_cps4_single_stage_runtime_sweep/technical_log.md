---
doc_type: technical_log
status: active
---

# OASIS cps4 Single-Stage Runtime Sweep Technical Log

## Known Limitations

- This is a first10 screening sweep, not the final full-population result.
- Screening uses full evaluation/output mode, so runtime includes evaluation,
  regularity metrics, dense save, and packaging. DVF-only persistent timing is
  deferred until after promising candidates are selected.
- Adaptive early stopping monitors the training objective only. It does not
  evaluate Dice, HD95, or dense regularity during training.
- Existing cps2 rows are reference anchors and are not rerun in this sweep.

## 2026-08-05 Plan

- Add single-stage adaptive early stopping to `run_one.py`.
- Add a cps4 launcher and aggregator for D-BSCP and SINR candidates.
- Run dry-run manifest, rank-0 smoke with `1` epoch, then full first10 across
  all visible GPUs.
- Select promising candidates using same-method cps4 baseline deltas and the
  runtime/epoch gate.

## 2026-08-06 Execution Summary

- Implemented single-stage adaptive early stopping in `run_one.py` for
  `direct_cp_cps4_be_bspline_analytic` and `sinr_cps4_be_bspline_analytic`.
  The monitor is total training loss per epoch; no extra validation/evaluation
  pass is added during training.
- Added generalized screening launcher and aggregator:
  `launch_oasis_cps4_runtime_sweep.py` and
  `aggregate_oasis_cps4_runtime_sweep.py`.
- Added follow-up persistent DVF-only timing runner and analyzer:
  `run_oasis_cps4_persistent_dvf_timing.py` and
  `analyze_oasis_cps4_persistent_dvf_timing.py`.
- Dry-run manifest produced the expected `120` full-screening jobs.
- Rank-0 smoke completed `12/12` screening jobs and aggregated without errors.
- Full first10 screening completed `120/120` jobs across GPUs `0,1,2,3`.
- Aggregation validation passed with zero missing summaries and zero reported
  aggregation errors.

## Full-Pipeline Screening Result

Baseline first10 rows in the same sweep:

| method | candidate | Dice | HD95 | total elapsed | epochs |
| --- | --- | ---: | ---: | ---: | ---: |
| D-BSCP cps4 | `D_baseline_lr1e-3_e2500` | 0.7965 | 2.3120 | 164.4 s | 2500.0 |
| SINR cps4 | `S_baseline_lr1e-4_e2500` | 0.7945 | 2.2745 | 191.9 s | 2500.0 |

Promising candidates selected for timing follow-up:

| method | candidate | Dice | HD95 | Dice delta | HD95 delta | total elapsed | saving | epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D-BSCP cps4 | `D_A_lr1e-2` | 0.7946 | 2.3653 | -0.0019 | +0.0533 | 105.4 s | 35.9% | 476.9 |
| D-BSCP cps4 | `D_F600_lr1e-2` | 0.7949 | 2.3795 | -0.0016 | +0.0676 | 117.7 s | 28.4% | 600.0 |
| D-BSCP cps4 | `D_F1000_lr3e-3` | 0.7958 | 2.3343 | -0.0007 | +0.0224 | 120.0 s | 27.0% | 1000.0 |
| SINR cps4 | `S_F1000_lr1e-3` | 0.7866 | 2.2503 | -0.0079 | -0.0242 | 131.0 s | 31.7% | 1000.0 |
| SINR cps4 | `S_A_lr1e-3` | 0.7813 | 2.2331 | -0.0132 | -0.0414 | 134.4 s | 29.9% | 846.4 |

Rejected candidates:

- `D_A_lr3e-2` and `D_F400_lr3e-2`: runtime was faster, but HD95 increase
  exceeded the `0.2 mm` tolerance.
- `S_A_lr3e-3` and `S_F600_lr3e-3`: high learning rate degraded Dice/HD95
  substantially.
- `S_F1500_lr3e-4`: accuracy was preserved, but full-pipeline runtime saving
  was below the `25%` gate.

None of the promising cps4 variants beat the same-family cps2 reference on both
Dice and HD95. The cps2 rows were comparison anchors only and were not rerun.

## Persistent DVF-Only Timing Follow-Up

- Timed baselines plus the five promising candidates.
- Process model: one long-lived Python worker per candidate, ranks `0-9`
  sequentially on GPU `0`.
- Cold case is rank `0`; warm cases are ranks `1-9` in the same process.
  Warm means process/GPU warm, not registration warm-start; each case uses a
  fresh image pair, model/control state, optimizer, and seed reset.
- Smoke persistent timing completed `14/14` jobs.
- Full persistent timing completed `70/70` jobs.
- Validation passed: dense DVF files `70/70`, timing JSONs `70/70`, failed
  subprocesses `0`, evaluation/full-output events in DVF-only timing `0`,
  non-target GPU observations `0`, reconciliation residual failures `0`, and
  parent self-unaccounted warnings `0`.

Warm persistent-process loaded-images-to-DVF-file means:

| method | candidate | loaded images -> DVF file | external wall | optimizer loop | epochs |
| --- | --- | ---: | ---: | ---: | ---: |
| D-BSCP cps4 | `D_baseline_lr1e-3_e2500` | 80.80 s | 81.29 s | 80.67 s | 2500.0 |
| D-BSCP cps4 | `D_F1000_lr3e-3` | 31.83 s | 32.32 s | 31.76 s | 1000.0 |
| D-BSCP cps4 | `D_F600_lr1e-2` | 19.13 s | 19.56 s | 19.05 s | 600.0 |
| D-BSCP cps4 | `D_A_lr1e-2` | 15.12 s | 15.56 s | 15.05 s | 473.2 |
| SINR cps4 | `S_baseline_lr1e-4_e2500` | 106.92 s | 107.34 s | 106.80 s | 2500.0 |
| SINR cps4 | `S_F1000_lr1e-3` | 42.73 s | 43.17 s | 42.63 s | 1000.0 |
| SINR cps4 | `S_A_lr1e-3` | 37.04 s | 37.47 s | 36.94 s | 867.7 |

Warm persistent-process savings versus the same-method cps4 baseline:

| candidate | loaded-images DVF-file saving | external-wall saving | epoch reduction |
| --- | ---: | ---: | ---: |
| `D_F1000_lr3e-3` | 60.6% | 60.2% | 60.0% |
| `D_F600_lr1e-2` | 76.3% | 75.9% | 76.0% |
| `D_A_lr1e-2` | 81.3% | 80.9% | 81.1% |
| `S_F1000_lr1e-3` | 60.0% | 59.8% | 60.0% |
| `S_A_lr1e-3` | 65.4% | 65.1% | 65.3% |

Primary review files:

- `method_ordered_report.md`
- `candidate_summary.csv`
- `accuracy_runtime_tradeoff.csv`
- `promising_candidates.csv`
- `persistent_cps4_dvf_timing_report.md`
- `persistent_cps4_summary.csv`
- `persistent_cps4_runtime_vs_baseline.csv`
