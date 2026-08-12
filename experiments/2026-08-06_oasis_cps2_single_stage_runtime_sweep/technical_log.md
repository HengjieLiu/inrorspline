---
doc_type: technical_log
status: active
---

# OASIS cps2 Single-Stage Runtime Sweep Technical Log

## Known Limitations

- This is a first10 screening sweep, not a final full-population result.
- Screening uses full evaluation/output mode, so runtime includes evaluation,
  regularity metrics, dense save, and packaging.
- Persistent timing is DVF-only and intentionally excludes evaluation and full
  packaging.
- Adaptive early stopping monitors the training objective only. It does not
  evaluate Dice, HD95, or dense regularity during training.
- Existing cps4 and MR-D-BSCP rows are contextual references, not primary
  gates.

## 2026-08-06 Plan

- Adapt the completed cps4 single-stage runtime workflow to cps2.
- Enable `direct_cp_cps2_be_bspline_analytic` in DVF-only mode.
- Run dry-run manifest, rank-0 smoke with `1` epoch, then full first10 across
  GPUs `0,1,2,3`.
- Select promising candidates using same-method cps2 baseline deltas and the
  runtime/epoch gate.
- Run sequential persistent-process DVF-only timing for promising candidates
  plus matching baselines on GPU `0`.

## 2026-08-07 Completion

- Syntax checks passed for `run_one.py` and the cps2 launch/aggregate/timing
  scripts.
- Dry-run manifest produced the expected `120` full-screening jobs.
- Full screening completed `120/120` jobs with zero missing summaries.
- Promising candidates selected:
  - D-BSCP cps2: `D2_A_lr3e-2`, `D2_A_lr1e-2`, `D2_F400_lr3e-2`,
    `D2_F600_lr1e-2`.
  - SINR cps2: `S2_F1000_lr1e-3`, `S2_A_lr1e-3`, `S2_F1500_lr3e-4`.
- Rejected notable near/misfit candidates:
  - `D2_F1000_lr3e-3` preserved accuracy but missed the 25% full-pipeline
    runtime-saving gate (`21.5%` saving).
  - `S2_F600_lr3e-3` and `S2_A_lr3e-3` were fast but degraded accuracy badly
    (`Dice` drop about `0.193`, `HD95` increase about `1.366 mm`).

## Screening Results

- D-BSCP cps2 baseline: `Dice=0.8023`, `HD95=2.2788 mm`,
  full-pipeline total `211.4 s/case`, `2500` epochs.
- Best D-BSCP runtime candidate by screening: `D2_A_lr3e-2`,
  `Dice=0.8004`, `HD95=2.4368 mm`, `214.3` epochs,
  full-pipeline total `133.4 s/case` (`36.9%` saving).
- Most accuracy-preserving D-BSCP speed candidate: `D2_A_lr1e-2`,
  `Dice=0.8022`, `HD95=2.3277 mm`, `379.7` epochs,
  full-pipeline total `145.6 s/case` (`31.1%` saving).
- SINR cps2 baseline: `Dice=0.7992`, `HD95=2.2374 mm`,
  full-pipeline total `402.5 s/case`, `2500` epochs.
- Best SINR runtime candidate by screening: `S2_F1000_lr1e-3`,
  `Dice=0.7918`, `HD95=2.2848 mm`, full-pipeline total
  `227.3 s/case` (`43.5%` saving).
- Most HD95-improving SINR speed candidate: `S2_F1500_lr3e-4`,
  `Dice=0.7948`, `HD95=2.1884 mm`, full-pipeline total
  `285.1 s/case` (`29.2%` saving).

## Persistent DVF-Only Timing

- Persistent timing completed for `9` candidates x `10` ranks = `90/90`
  cases.
- Validation after rerunning the D-BSCP cps2 baseline under clean GPU
  conditions:
  - dense DVF files: `90/90`;
  - failed subprocesses: `0`;
  - DVF-only evaluation/full-output events: `0`;
  - reconciliation residual failures: `0`;
  - parent self-unaccounted warnings: `0`;
  - non-target GPU compute-app observations: `0`.
- Warm ranks `1-9`, external wall per case:
  - D-BSCP cps2 baseline: `86.00 s`; `D2_A_lr3e-2`: `8.82 s`
    (`89.7%` saving); `D2_A_lr1e-2`: `14.67 s` (`82.9%` saving);
    `D2_F400_lr3e-2`: `15.41 s` (`82.1%` saving);
    `D2_F600_lr1e-2`: `21.33 s` (`75.2%` saving).
  - SINR cps2 baseline: `273.91 s`; `S2_F1000_lr1e-3`: `109.39 s`
    (`60.1%` saving); `S2_A_lr1e-3`: `114.34 s` (`58.3%` saving);
    `S2_F1500_lr3e-4`: `166.99 s` (`39.0%` saving).
- Primary reports:
  - `method_ordered_report.md`
  - `persistent_cps2_dvf_timing_report.md`
  - `persistent_cps2_screen_and_timing.csv`
