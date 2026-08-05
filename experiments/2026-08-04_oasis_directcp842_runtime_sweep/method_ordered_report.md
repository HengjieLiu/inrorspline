# OASIS DirectCP 8-4-2 Runtime Sweep: Ordered Method Report

Generated from the first10 aggregate artifacts for the DirectCP `8-4-2`
runtime sweep.

## Scope

- Dataset/protocol: OASIS BrainMR seed-3 shuffled ranks `0-9`.
- Runtime-sweep method family:
  `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- Paper-facing method name: `MR-D-BSCP`, implemented here as direct
  multiresolution cubic B-spline control-point optimization with exact
  crop-aligned dyadic knot insertion.
- Regularity setting: alpha `1000`.
- Ordered by mean total runtime, fastest first.
- Performance columns are first10 population means. `dDice` and `dHD95` are
  differences against the original MR-D-BSCP / DirectCP `8-4-2` reference row.

## Original Reference

The reference row is the original MR-D-BSCP / DirectCP `8-4-2` setting used for
the matched comparison:

- Stage cps: `8,4,2`.
- Stage downsample: `4,2,1`.
- Epoch schedule: `400,800,1300` (`2500` total).
- LR: `1e-3`.
- Alpha: `1000`.
- Seed: `1`.
- Baseline source:
  `experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization/summary_runs_combined.csv`.

## Ordered Candidate Table

| order | method | description | lr | planned epochs | actual epochs | stopping | Dice | dDice | HD95 | dHD95 | train s | total s | gap s | speedup total |
| ---: | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | Original MR-D-BSCP / DirectCP 8-4-2 | fixed baseline, alpha 1000, lr 1e-3 | 0.001 | 400,800,1300 | 2500.0 | none | 0.8070 | 0.0000 | 2.2020 | 0.0000 | 194.8 | 229.7 | 35.0 | 1.00x |
| 1 | A_fast_decay | adaptive fast plateau, decayed stage LR; stage lr 3e-2,1e-2,3e-3 | 0.01 | 400,800,1300 | 577.5 | adaptive min 75,150,250; pat 50,75,100 | 0.8058 | -0.0012 | 2.2147 | +0.0127 | 149.5 | 183.2 | 33.7 | 1.25x |
| 2 | F600_decay | fixed 600 epochs, decayed stage LR; stage lr 3e-2,1e-2,3e-3 | 0.01 | 100,200,300 | 600.0 | fixed | 0.8063 | -0.0007 | 2.2089 | +0.0069 | 151.8 | 186.2 | 34.4 | 1.23x |
| 3 | F600_lr1e-2 | fixed 600 epochs | 0.01 | 100,200,300 | 600.0 | fixed | 0.8059 | -0.0011 | 2.2312 | +0.0292 | 152.7 | 186.3 | 33.6 | 1.23x |
| 4 | F600_lr3e-2 | fixed 600 epochs | 0.03 | 100,200,300 | 600.0 | fixed | 0.8041 | -0.0029 | 2.2691 | +0.0671 | 154.0 | 188.7 | 34.7 | 1.22x |
| 5 | A_bal_lr1e-2 | adaptive balanced plateau | 0.01 | 400,800,1300 | 720.0 | adaptive min 100,200,300; pat 75,100,150 | 0.8061 | -0.0009 | 2.2147 | +0.0127 | 154.7 | 190.0 | 35.3 | 1.21x |
| 6 | A_bal_lr3e-2 | adaptive balanced plateau | 0.03 | 400,800,1300 | 702.5 | adaptive min 100,200,300; pat 75,100,150 | 0.8046 | -0.0024 | 2.2641 | +0.0621 | 156.8 | 190.6 | 33.8 | 1.21x |
| 7 | F950_lr1e-2 | fixed 950 epochs | 0.01 | 150,300,500 | 950.0 | fixed | 0.8061 | -0.0008 | 2.2244 | +0.0224 | 157.9 | 193.3 | 35.3 | 1.19x |
| 8 | F950_lr3e-2 | fixed 950 epochs | 0.03 | 150,300,500 | 950.0 | fixed | 0.8047 | -0.0023 | 2.2572 | +0.0552 | 157.9 | 193.7 | 35.8 | 1.19x |
| 9 | A_safe_lr3e-2 | adaptive conservative plateau | 0.03 | 400,800,1300 | 1062.5 | adaptive min 150,300,500; pat 100,150,200 | 0.8036 | -0.0033 | 2.2567 | +0.0547 | 160.6 | 194.3 | 33.8 | 1.18x |
| 10 | F950_decay | fixed 950 epochs, decayed stage LR; stage lr 3e-2,1e-2,3e-3 | 0.01 | 150,300,500 | 950.0 | fixed | 0.8057 | -0.0013 | 2.2186 | +0.0166 | 160.7 | 194.6 | 33.9 | 1.18x |
| 11 | A_safe_lr1e-2 | adaptive conservative plateau | 0.01 | 400,800,1300 | 1135.0 | adaptive min 150,300,500; pat 100,150,200 | 0.8054 | -0.0016 | 2.2445 | +0.0425 | 163.2 | 198.6 | 35.3 | 1.16x |
| 12 | F1550_lr3e-3 | fixed 1550 epochs | 0.003 | 250,500,800 | 1550.0 | fixed | 0.8071 | +0.0002 | 2.2124 | +0.0104 | 172.1 | 205.8 | 33.7 | 1.12x |
| 13 | F1550_lr3e-2 | fixed 1550 epochs | 0.03 | 250,500,800 | 1550.0 | fixed | 0.8039 | -0.0031 | 2.2644 | +0.0624 | 176.1 | 208.8 | 32.6 | 1.10x |
| 14 | F1550_lr1e-2 | fixed 1550 epochs | 0.01 | 250,500,800 | 1550.0 | fixed | 0.8056 | -0.0014 | 2.2329 | +0.0309 | 180.3 | 214.5 | 34.2 | 1.07x |

## Readout

- Fastest candidate: `A_fast_decay`, total `183.2 s`, train `149.5 s`,
  Dice `0.8058`, HD95 `2.2147`.
- Best fixed short schedule: `F600_decay`, total `186.2 s`, train `151.8 s`,
  Dice `0.8063`, HD95 `2.2089`.
- Highest Dice among tested candidates: `F1550_lr3e-3`, Dice `0.8071`, total
  `205.8 s`.
- The training-vs-total gap is nearly fixed at about `33-35 s`, so the runtime
  improvement comes primarily from reducing optimization epochs.

## Source Artifacts

- First10 review report:
  `experiments/2026-08-04_oasis_directcp842_runtime_sweep/first10/review_report.md`.
- Candidate descriptions and epoch settings:
  `experiments/2026-08-04_oasis_directcp842_runtime_sweep/first10/candidate_population_summary.csv`.
- Candidate acceptance/performance:
  `experiments/2026-08-04_oasis_directcp842_runtime_sweep/first10/candidate_acceptance.csv`.
- Original MR-D-BSCP / DirectCP `8-4-2` reference:
  `experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization/summary_runs_combined.csv`.
