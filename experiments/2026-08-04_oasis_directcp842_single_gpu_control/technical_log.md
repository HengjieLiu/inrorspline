# Technical Log: OASIS DirectCP 8-4-2 Single-GPU Runtime Control

## Known Limitations

- This is a single-case timing control on OASIS seed-3 rank `0`, not a final
  first10 or population claim.
- The goal is to isolate serial single-GPU timing from the earlier P17
  four-GPU concurrent screening run.
- Total runtime includes dense evaluation, label metrics, regularity metrics,
  serialization, and command overhead. Training time is the primary optimizer
  runtime.

## Planned Runs

- Original MR-D-BSCP / DirectCP `8-4-2`: lr `1e-3`, stage epochs
  `400,800,1300`, no early stopping.
- `A_fast_decay`: base lr `1e-2`, stage lr `3e-2,1e-2,3e-3`, adaptive stopping
  with max stage epochs `400,800,1300`, minimums `75,150,250`, patience
  `50,75,100`.

## 2026-08-04 Setup

- Docker container: `py310sitreg_v1_test1`.
- GPU policy: run serially on GPU `0`; leave GPUs `1-3` idle.
- Pair: OASIS seed-3 shuffled rank `0`, fixed `0333`, moving `0115`, original
  sorted index `158`.
- Prelaunch GPU state: all four RTX 6000 Ada GPUs at `5 MiB` and `0%`
  utilization; no active compute processes.

## 2026-08-04 Result

- Original MR-D-BSCP / DirectCP `8-4-2` completed with `DONE`.
  - Actual epochs: `2500`, stages `[400,800,1300]`.
  - Stage elapsed seconds: `[36.64, 29.60, 63.39]`.
  - Training time: `186.69 s`.
  - Total script time: `216.20 s`.
  - Command wall-clock time: `226.14 s`.
  - Dice / HD95: `0.76074` / `2.65234`.
- `A_fast_decay` completed with `DONE`.
  - Actual epochs: `600`, stages `[150,200,250]`.
  - Stage elapsed seconds: `[34.09, 19.29, 31.73]`.
  - Training time: `151.08 s`.
  - Total script time: `181.89 s`.
  - Command wall-clock time: `191.18 s`.
  - Dice / HD95: `0.75901` / `2.66425`.
- Serial single-GPU speedup:
  - Epoch ratio: `2500/600 = 4.17x` fewer epochs for `A_fast_decay`.
  - Training speedup: `1.24x` (`35.61 s` saved).
  - Total script speedup: `1.19x` (`34.31 s` saved).
  - Command wall-clock speedup: `1.18x` (`34.96 s` saved).
- Interpretation: the modest wall-clock speedup is not primarily from four-GPU
  concurrency in the P17 sweep. On this rank, large fixed costs remain inside
  stage setup/prolongation/full-resolution evaluation and post-training dense
  metrics/serialization. Coarse-stage epoch reductions especially do not
  translate linearly into time savings.
