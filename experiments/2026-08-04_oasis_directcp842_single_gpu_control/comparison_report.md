# OASIS DirectCP 8-4-2 Single-GPU Runtime Control

This control runs the original MR-D-BSCP / DirectCP `8-4-2` reference and
`A_fast_decay` serially on OASIS seed-3 rank `0`, GPU `0` only, with GPUs `1-3`
idle.

## Result

| method | epochs | stage epochs | train s | total s | wall s | Dice | HD95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Original MR-D-BSCP / DirectCP 8-4-2 | 2500 | 400,800,1300 | 186.69 | 216.20 | 226.14 | 0.76074 | 2.65234 |
| A_fast_decay | 600 | 150,200,250 | 151.08 | 181.89 | 191.18 | 0.75901 | 2.66425 |

## Delta

| comparison | value |
| --- | ---: |
| Epoch reduction | 4.17x fewer epochs |
| Train time saved | 35.61 s |
| Total script time saved | 34.31 s |
| Command wall time saved | 34.96 s |
| Train speedup | 1.24x |
| Total script speedup | 1.19x |
| Command wall speedup | 1.18x |
| Dice change | -0.00173 |
| HD95 change | +0.01190 mm |

## Interpretation

The controlled run supports that `A_fast_decay` is genuinely faster under
serial one-GPU execution, but the speedup is not proportional to epoch count.
The rank-0 epoch count drops from `2500` to `600`, yet wall time drops only from
`226.14 s` to `191.18 s`.

The reason is visible in the stage timings:

| method | stage 0 s | stage 1 s | stage 2 s | stage sum s |
| --- | ---: | ---: | ---: | ---: |
| Original | 36.64 | 29.60 | 63.39 | 129.63 |
| A_fast_decay | 34.09 | 19.29 | 31.73 | 85.10 |

Large fixed costs remain outside pure optimizer-loop epoch count, including
stage setup, prolongation, dense evaluation, regularity/label metrics, and
serialization. Coarse-stage epoch reductions are especially non-linear in wall
time: stage 0 drops from `400` to `150` epochs but only from `36.64 s` to
`34.09 s`.

## Decision Note

If the goal is only to test whether the P17 speedup was an artifact of running
four jobs concurrently, this single-rank control is enough: it was not mainly a
four-GPU contention artifact. If the goal is a review-facing runtime table with
controlled serial timing, the next clean step is to run the same two methods on
first10 serially on one GPU. Based on rank 0, that should take roughly
`65-70 minutes`.
