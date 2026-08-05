# Technical Log: OASIS DirectCP 8-4-2 DVF-Only Timing

## Known Limitations

- Rank-0 timing characterization only.
- Deep timing instrumentation may add small overhead, but both compared methods
  use the same timing mode.
- DVF-only mode intentionally excludes Dice, HD95/ASD, dense NCC, regularity
  metrics, checkpoint saves, and full CSV packaging. The report must not be
  interpreted as an accuracy or full-pipeline runtime table.
- `dvf_in_memory_ready` is marked immediately after final-stage dense
  displacement expansion. Final control-summary stats occur after that mark and
  are part of total non-loop training, but not part of the in-memory DVF
  endpoint.

## Launch Notes

- Date: 2026-08-04.
- Container: `py310sitreg_v1_test1`.
- GPU policy: serial subprocesses on GPU `0`; GPUs `1-3` intentionally idle.
- Output root:
  `<external-run-root>/runs/oasis_directcp842_dvf_only_timing_alpha1000_v1`.

## Commands

Smoke:

```bash
python scripts/experiments/oasis_alpha_sweep/run_directcp842_dvf_only_timing.py \
  --phase smoke \
  --gpu 0 \
  --overwrite
```

Rank-0:

```bash
python scripts/experiments/oasis_alpha_sweep/run_directcp842_dvf_only_timing.py \
  --phase rank0 \
  --gpu 0 \
  --overwrite
```

## Results Log

- `py_compile` passed for `run_one.py`,
  `run_directcp842_dvf_only_timing.py`, and
  `analyze_directcp842_dvf_only_timing.py` after redirecting bytecode output to
  `/tmp/inrorspline_pycompile`.
- Smoke phase completed `2/2` jobs with `DONE`, `summary.json`,
  `timing.json`, and dense DVF `.npy` files present. Smoke analyzer reported
  evaluation/full-output events `0`, reconciliation residual failures `0`, and
  parent self-unaccounted warnings `0`.
- Rank-0 phase completed `2/2` jobs with dense DVF files present:
  `disp_norm_0333_0115.npy` for both methods.
- Rank-0 analyzer reported evaluation/full-output events `0`, reconciliation
  residual failures `0`, parent self-unaccounted warnings `0`, and non-target
  GPU compute observations `0`.

## Rank-0 Headline

| method | epochs | full script -> DVF memory | loaded images -> DVF memory | full script -> DVF file | loaded images -> DVF file | external wall |
|---|---:|---:|---:|---:|---:|---:|
| original MR-D-BSCP / DirectCP `8-4-2` | 2500 | 234.18 s | 206.51 s | 234.26 s | 206.60 s | 238.22 s |
| `A_fast_decay` | 600 | 175.22 s | 146.77 s | 175.28 s | 146.83 s | 178.99 s |

Savings are `original - A_fast_decay`:

- In-memory DVF: `58.96 s`.
- DVF file ready: `58.98 s`.
- Command wall: `59.24 s`.
- Optimizer loop: `52.20 s`.
- Non-loop training: `8.41 s`.
- Dense DVF file write itself: about `0.03-0.04 s` for both methods.

Primary review surface:

- `dvf_only_timing_report.md`
