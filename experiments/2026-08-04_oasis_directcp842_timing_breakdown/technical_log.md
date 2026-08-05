# Technical Log

## Known Limitations

- Deep timing uses synchronized parent intervals plus CUDA event summaries
  inside the optimizer loop. The event collection can add small overhead, so
  compare original and `A_fast_decay` only under the same timing mode.
- `external_wall_sec` includes Python startup and import time outside
  `run_one.py`'s script timer. This is reported as external uninstrumented
  time rather than assigned to a training category.
- The first pass is rank `0` only. First10 serial timing should run only after
  the rank-0 `timing.json` reconciliation and analyzer outputs are acceptable.
- Output saving remains enabled so measured time reflects the current practical
  pipeline rather than a stripped training-only benchmark.

## 2026-08-04

- Added optional `--timing-detail {off,coarse,deep}` instrumentation to the
  DirectCP multistage path in `run_one.py`.
- Added serial one-GPU launcher:
  `scripts/experiments/oasis_alpha_sweep/run_directcp842_timing_breakdown.py`.
- Added analyzer:
  `scripts/experiments/oasis_alpha_sweep/analyze_directcp842_timing_breakdown.py`.
- Planned execution order: tiny smoke, rank-0 serial pair, analyzer check,
  first10 serial pair if reconciliation passes.
- `py_compile` passed for the modified runner, timing launcher, and analyzer.
- Tiny smoke passed for both candidates with `1,1,1` epochs. The smoke already
  exposed large fixed costs from prolongation, final dense expansion, dense
  metrics, regularity metrics, and serialization.
- Rank-0 serial pair passed with residual failures `0`, parent intervals with
  large self-unaccounted time `0`, and no non-target GPU compute observations.
  Rank-0 wall/script/training/optimizer-loop:
  - Original MR-D-BSCP / DirectCP `8-4-2`: `233.67/230.19/191.22/107.07 s`
  - `A_fast_decay`: `186.72/183.30/145.04/60.76 s`
- First10 serial timing passed with `20/20` jobs complete, `20/20`
  `timing.json` files, failed subprocesses `0`, residual failures `0`, parent
  intervals with large self-unaccounted time `0`, external uninstrumented wall
  mean `4.29 s`, and no non-target GPU compute observations.
- First10 means:
  - Original MR-D-BSCP / DirectCP `8-4-2`: wall `247.76 s`, script
    `243.47 s`, training `206.12 s`, optimizer loop `110.59 s`, fixed/io known
    `132.10 s`, epochs `2500`, Dice `0.80697`, HD95 `2.2020`.
  - `A_fast_decay`: wall `206.62 s`, script `202.32 s`, training `161.79 s`,
    optimizer loop `63.69 s`, fixed/io known `137.80 s`, epochs `577.5`, Dice
    `0.80581`, HD95 `2.2147`.
- First10 paired mean savings from original to `A_fast_decay`: wall
  `41.14 s`, script `41.15 s`, optimizer loop `46.89 s`, epoch reduction
  `1922.5`, Dice delta `-0.00116`, HD95 delta `+0.0127 mm`.
- Primary interpretation: most epoch savings land in the optimizer-loop
  bucket, but the practical wall-clock gain is capped by fixed setup,
  prolongation/final-expansion, dense evaluation, regularity metrics, and
  serialization costs. Under detailed timing, fixed/io known time alone is
  about `132-138 s` per case.
