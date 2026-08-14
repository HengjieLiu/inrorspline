# OASIS Runtime Reduction Selection Summary

Generated: 2026-08-12

## Purpose

This note summarizes the completed OASIS seed-3 first10 runtime-reduction
experiments and supports selecting one accelerated version each for future
first100 validation:

- MR-D-BSCP `8-4-2`
- D-BSCP `cps2`
- SINR `cps2`

No first100 run has been launched from this summary. The next full evaluation
should wait until the accelerated variants are explicitly selected.

## Source Experiments

- `experiments/2026-08-04_oasis_directcp842_runtime_sweep/`
- `experiments/2026-08-04_oasis_directcp842_persistent_process_dvf_timing/`
- `experiments/2026-08-05_oasis_cps4_single_stage_runtime_sweep/`
- `experiments/2026-08-06_oasis_cps2_single_stage_runtime_sweep/`

The screening tables report full-pipeline first10 runtime and accuracy. The
persistent DVF timing tables report production DVF-only time from loaded input
images to the dense DVF saved to disk, excluding Dice, HD95/ASD, dense NCC,
regularity metrics, checkpoint saves, and full output packaging.

## Current Recommended Selections

| Target method | Recommended accelerated candidate | Rationale |
|---|---|---|
| MR-D-BSCP `8-4-2` | `A_fast_decay` | Fastest accepted first10 candidate; Dice drop only `0.0012`, HD95 increase only `0.0127 mm`; warm DVF-to-disk time drops from `62.15 s` to `14.63 s`. |
| D-BSCP `cps2` | `D2_A_lr1e-2` | Conservative balance; almost no Dice loss (`-0.0001`) and modest HD95 increase (`+0.0489 mm`) with `31.1%` full-pipeline saving. |
| D-BSCP `cps2`, speed-priority alternative | `D2_A_lr3e-2` | Larger full-pipeline saving (`36.9%`) and very fast warm DVF timing (`7.95 s`), but HD95 increase is larger (`+0.1580 mm`). |
| SINR `cps2` | `S2_F1000_lr1e-3` | Best speed/accuracy balance among SINR cps2; `43.5%` full-pipeline saving with Dice drop `0.0074` and HD95 increase `0.0474 mm`. |

Recommended default for a clean first100 validation set:

```text
MR-D-BSCP 8-4-2: A_fast_decay
D-BSCP cps2:     D2_A_lr1e-2
SINR cps2:       S2_F1000_lr1e-3
```

Use `D2_A_lr3e-2` only if the revision prioritizes maximum runtime reduction
over the most conservative HD95 preservation.

## MR-D-BSCP 8-4-2 Screening

Reference method: original MR-D-BSCP / DirectCP `8,4,2`, alpha `1000`, stage
LR `1e-3`, stage epochs `400,800,1300`, no early stopping.

| candidate | schedule | actual epochs | Dice | dDice | HD95 | dHD95 | ASD | full total s | full speedup | peak GPU GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | lr `1e-3`; `400,800,1300`; no stop | 2500.0 | 0.8070 | 0.0000 | 2.2020 | 0.0000 | NA | 229.7 | 1.00x | 3.93 |
| `A_fast_decay` | stage lr `3e-2,1e-2,3e-3`; adaptive min `75,150,250`; patience `50,75,100` | 577.5 | 0.8058 | -0.0012 | 2.2147 | +0.0127 | 0.4445 | 183.2 | 1.25x | 3.93 |
| `F600_decay` | stage lr `3e-2,1e-2,3e-3`; fixed `100,200,300` | 600.0 | 0.8063 | -0.0007 | 2.2089 | +0.0069 | 0.4433 | 186.2 | 1.23x | 3.93 |
| `A_bal_lr1e-2` | lr `1e-2`; adaptive min `100,200,300`; patience `75,100,150` | 720.0 | 0.8061 | -0.0009 | 2.2147 | +0.0127 | 0.4461 | 190.0 | 1.21x | 3.93 |
| `F1550_lr3e-3` | lr `3e-3`; fixed `250,500,800` | 1550.0 | 0.8071 | +0.0002 | 2.2124 | +0.0104 | 0.4422 | 205.8 | 1.12x | 3.93 |

Full candidate order is in
`experiments/2026-08-04_oasis_directcp842_runtime_sweep/method_ordered_report.md`.

### MR-D-BSCP DVF-Only Persistent Timing

Loaded input images to dense DVF file on disk:

| candidate | cold rank0 s | warm ranks1-9 s | amortized ranks0-9 s | optimizer warm s | epochs warm |
|---|---:|---:|---:|---:|---:|
| original | 147.77 | 62.15 | 70.71 | 62.03 | 2500.0 |
| `A_fast_decay` | 117.01 | 14.63 | 24.87 | 14.50 | 575.0 |

Warm persistent-process saving from original to `A_fast_decay`:

```text
loaded-images-to-DVF-file saving = 62.15 - 14.63 = 47.52 s/case
relative saving = 76.5%
```

The large cold/warm gap is not registration warm-start. Each case has fresh
registration parameters and optimizer state. The warm speedup comes from a
long-lived Python/GPU process: CUDA context, PyTorch allocator state, kernel
first-use state, and algorithm/cache reuse for the same OASIS tensor shapes.

## D-BSCP cps4 Screening Context

These cps4 rows were tested mainly as a single-stage reference and as context
for the cps2 selection.

| candidate | schedule | actual epochs | Dice | dDice | HD95 | dHD95 | ASD | full total s | saving | peak GPU GB | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `D_baseline_lr1e-3_e2500` | lr `1e-3`; fixed `2500` | 2500.0 | 0.7965 | 0.0000 | 2.3120 | 0.0000 | 0.4672 | 164.4 | 0.0% | 1.75 | baseline |
| `D_A_lr1e-2` | lr `1e-2`; adaptive min `300`; patience `150` | 476.9 | 0.7946 | -0.0019 | 2.3653 | +0.0533 | 0.4761 | 105.4 | 35.9% | 1.75 | promising |
| `D_F600_lr1e-2` | lr `1e-2`; fixed `600` | 600.0 | 0.7949 | -0.0016 | 2.3795 | +0.0676 | 0.4778 | 117.7 | 28.4% | 1.75 | promising |
| `D_F1000_lr3e-3` | lr `3e-3`; fixed `1000` | 1000.0 | 0.7958 | -0.0007 | 2.3343 | +0.0224 | 0.4728 | 120.0 | 27.0% | 1.75 | promising |
| `D_A_lr3e-2` | lr `3e-2`; adaptive min `200`; patience `100` | 326.3 | 0.7889 | -0.0076 | 2.6665 | +0.3545 | 0.5214 | 94.2 | 42.7% | 1.75 | rejected, HD95 |
| `D_F400_lr3e-2` | lr `3e-2`; fixed `400` | 400.0 | 0.7884 | -0.0081 | 2.6447 | +0.3328 | 0.5208 | 100.0 | 39.1% | 1.75 | rejected, HD95 |

DVF-only loaded-images-to-DVF-file timing:

| candidate | cold rank0 s | warm ranks1-9 s | amortized ranks0-9 s |
|---|---:|---:|---:|
| `D_baseline_lr1e-3_e2500` | 131.08 | 80.80 | 85.83 |
| `D_F1000_lr3e-3` | 81.94 | 31.83 | 36.84 |
| `D_F600_lr1e-2` | 62.38 | 19.13 | 23.45 |
| `D_A_lr1e-2` | 58.47 | 15.12 | 19.45 |

## D-BSCP cps2 Screening

Reference method: `direct_cp_cps2_be_bspline_analytic`, alpha `1000`, lr
`1e-3`, fixed `2500` epochs.

| candidate | schedule | actual epochs | Dice | dDice | HD95 | dHD95 | ASD | full total s | saving | peak GPU GB | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `D2_baseline_lr1e-3_e2500` | lr `1e-3`; fixed `2500` | 2500.0 | 0.8023 | 0.0000 | 2.2788 | 0.0000 | 0.4637 | 211.4 | 0.0% | 2.09 | baseline |
| `D2_A_lr3e-2` | lr `3e-2`; adaptive min `200`; patience `100` | 214.3 | 0.8004 | -0.0020 | 2.4368 | +0.1580 | 0.4758 | 133.4 | 36.9% | 2.09 | promising, speed-priority |
| `D2_A_lr1e-2` | lr `1e-2`; adaptive min `300`; patience `150` | 379.7 | 0.8022 | -0.0001 | 2.3277 | +0.0489 | 0.4659 | 145.6 | 31.1% | 2.09 | recommended |
| `D2_F400_lr3e-2` | lr `3e-2`; fixed `400` | 400.0 | 0.7998 | -0.0025 | 2.4437 | +0.1650 | 0.4769 | 150.8 | 28.7% | 2.09 | promising |
| `D2_F600_lr1e-2` | lr `1e-2`; fixed `600` | 600.0 | 0.8022 | -0.0002 | 2.3347 | +0.0560 | 0.4666 | 157.7 | 25.4% | 2.09 | promising |
| `D2_F1000_lr3e-3` | lr `3e-3`; fixed `1000` | 1000.0 | 0.8020 | -0.0004 | 2.2976 | +0.0188 | 0.4674 | 165.9 | 21.5% | 2.09 | accurate but below runtime gate |

DVF-only loaded-images-to-DVF-file timing:

| candidate | cold rank0 s | warm ranks1-9 s | amortized ranks0-9 s | optimizer warm s | epochs warm |
|---|---:|---:|---:|---:|---:|
| `D2_baseline_lr1e-3_e2500` | 156.86 | 85.24 | 92.40 | 85.01 | 2500.0 |
| `D2_A_lr3e-2` | 76.96 | 7.95 | 14.85 | 7.79 | 215.9 |
| `D2_A_lr1e-2` | 80.39 | 13.71 | 20.38 | 13.55 | 385.1 |
| `D2_F400_lr3e-2` | 83.67 | 14.21 | 21.16 | 14.05 | 400.0 |
| `D2_F600_lr1e-2` | 87.34 | 20.68 | 27.34 | 20.54 | 600.0 |

Interpretation: `D2_A_lr1e-2` is the cleanest paper candidate because it
almost exactly preserves Dice and has a small HD95 increase. `D2_A_lr3e-2` is
the strongest runtime candidate but has a larger HD95 increase, though still
within the predeclared `0.2 mm` gate.

## SINR cps4 Screening Context

These cps4 rows provide context for the cps2 SINR selection.

| candidate | schedule | actual epochs | Dice | dDice | HD95 | dHD95 | ASD | full total s | saving | peak GPU GB | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `S_baseline_lr1e-4_e2500` | lr `1e-4`; fixed `2500` | 2500.0 | 0.7945 | 0.0000 | 2.2745 | 0.0000 | 0.4695 | 191.9 | 0.0% | 2.50 | baseline |
| `S_F1000_lr1e-3` | lr `1e-3`; fixed `1000` | 1000.0 | 0.7866 | -0.0079 | 2.2503 | -0.0242 | 0.4798 | 131.0 | 31.7% | 2.50 | promising |
| `S_A_lr1e-3` | lr `1e-3`; adaptive min `500`; patience `200` | 846.4 | 0.7813 | -0.0132 | 2.2331 | -0.0414 | 0.4988 | 134.4 | 29.9% | 2.50 | promising but larger Dice loss |
| `S_F1500_lr3e-4` | lr `3e-4`; fixed `1500` | 1500.0 | 0.7943 | -0.0002 | 2.1781 | -0.0963 | 0.4618 | 148.4 | 22.7% | 2.50 | accurate but below runtime gate |
| `S_A_lr3e-3` | lr `3e-3`; adaptive min `350`; patience `150` | 350.0 | 0.5660 | -0.2285 | 3.8791 | +1.6046 | 1.3157 | 109.0 | 43.2% | 2.50 | rejected |
| `S_F600_lr3e-3` | lr `3e-3`; fixed `600` | 600.0 | 0.5640 | -0.2305 | 3.8803 | +1.6058 | 1.3224 | 121.3 | 36.8% | 2.50 | rejected |

DVF-only loaded-images-to-DVF-file timing:

| candidate | cold rank0 s | warm ranks1-9 s | amortized ranks0-9 s |
|---|---:|---:|---:|
| `S_baseline_lr1e-4_e2500` | 158.45 | 106.92 | 112.07 |
| `S_F1000_lr1e-3` | 98.87 | 42.73 | 48.34 |
| `S_A_lr1e-3` | 75.31 | 37.04 | 40.86 |

## SINR cps2 Screening

Reference method: `sinr_cps2_be_bspline_analytic`, alpha `1000`, lr `1e-4`,
fixed `2500` epochs.

| candidate | schedule | actual epochs | Dice | dDice | HD95 | dHD95 | ASD | full total s | saving | peak GPU GB | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `S2_baseline_lr1e-4_e2500` | lr `1e-4`; fixed `2500` | 2500.0 | 0.7992 | 0.0000 | 2.2374 | 0.0000 | 0.4535 | 402.5 | 0.0% | 7.63 | baseline |
| `S2_F1000_lr1e-3` | lr `1e-3`; fixed `1000` | 1000.0 | 0.7918 | -0.0074 | 2.2848 | +0.0474 | 0.4697 | 227.3 | 43.5% | 7.63 | recommended |
| `S2_A_lr1e-3` | lr `1e-3`; adaptive min `500`; patience `200` | 1041.7 | 0.7906 | -0.0087 | 2.2449 | +0.0074 | 0.4728 | 241.5 | 40.0% | 7.63 | promising |
| `S2_F1500_lr3e-4` | lr `3e-4`; fixed `1500` | 1500.0 | 0.7948 | -0.0045 | 2.1884 | -0.0491 | 0.4578 | 285.1 | 29.2% | 7.63 | slower, very safe HD95 |
| `S2_A_lr3e-3` | lr `3e-3`; adaptive min `350`; patience `150` | 495.2 | 0.6070 | -0.1922 | 3.6031 | +1.3657 | 1.1554 | 180.0 | 55.3% | 7.63 | rejected |
| `S2_F600_lr3e-3` | lr `3e-3`; fixed `600` | 600.0 | 0.6057 | -0.1935 | 3.6031 | +1.3656 | 1.1581 | 184.5 | 54.2% | 7.63 | rejected |

DVF-only loaded-images-to-DVF-file timing:

| candidate | cold rank0 s | warm ranks1-9 s | amortized ranks0-9 s | optimizer warm s | epochs warm |
|---|---:|---:|---:|---:|---:|
| `S2_baseline_lr1e-4_e2500` | 350.34 | 273.24 | 280.95 | 273.05 | 2500.0 |
| `S2_F1000_lr1e-3` | 161.33 | 108.75 | 114.01 | 108.59 | 1000.0 |
| `S2_A_lr1e-3` | 185.44 | 113.61 | 120.79 | 113.47 | 1044.3 |
| `S2_F1500_lr3e-4` | 245.26 | 164.50 | 172.57 | 164.32 | 1500.0 |

Interpretation: `S2_F1000_lr1e-3` is the best single SINR cps2 candidate for
first100 because it gives the largest accepted full-pipeline speedup without
the severe accuracy collapse seen at lr `3e-3`. `S2_F1500_lr3e-4` is a safer
accuracy candidate, but it gives less runtime reduction.

## Cross-Method Interpretation

1. MR-D-BSCP `8-4-2` benefits from two changes: a shorter adaptive schedule and
   skipping intermediate diagnostic dense expansions in production DVF timing.
   The final dense DVF expansion remains mandatory.
2. D-BSCP cps2 benefits strongly from higher LR plus adaptive stopping. The
   conservative choice `D2_A_lr1e-2` nearly preserves Dice, while the more
   aggressive `D2_A_lr3e-2` is much faster but degrades HD95 more.
3. SINR cps2 is sensitive to too-high LR. The `1e-3` variants remain acceptable;
   the `3e-3` variants are not usable because Dice and HD95 collapse.
4. Cold and warm timing must be reported separately. The cold case includes
   CUDA/PyTorch/library first-use overhead and is not comparable to warm
   persistent-process throughput as a pure per-case optimization cost.
5. Peak memory differs materially by method: MR-D-BSCP `8-4-2` is about
   `3.93 GB`, D-BSCP cps2 about `2.09 GB`, SINR cps2 about `7.63 GB` in the
   first10 screening runs.

## Future First100 Reporting Requirements

For each selected accelerated candidate and its full-budget counterpart, report:

- Dice, HD95, ASD, and regularity metrics on OASIS seed-3 ranks `0-99`.
- Full-pipeline runtime with evaluation and normal output behavior.
- Production DVF-only runtime from loaded input images to dense DVF saved to
  disk.
- Cold rank0 timing and warm ranks1-99 timing separately.
- Peak GPU memory per case, summarized as mean, standard deviation, maximum,
  and worst-case case ID.

The full-budget counterpart rows should be:

| accelerated target | full-budget counterpart |
|---|---|
| MR-D-BSCP `8-4-2` accelerated | original MR-D-BSCP `8-4-2`, `400,800,1300`, lr `1e-3` |
| D-BSCP cps2 accelerated | `D2_baseline_lr1e-3_e2500` |
| SINR cps2 accelerated | `S2_baseline_lr1e-4_e2500` |
