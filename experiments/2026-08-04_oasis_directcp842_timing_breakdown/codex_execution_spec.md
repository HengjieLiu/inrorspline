# OASIS DirectCP 8-4-2 Detailed Timing Breakdown

## Objective

Instrument and run a controlled one-GPU timing analysis for OASIS DirectCP
`8-4-2`, comparing the original MR-D-BSCP schedule with `A_fast_decay`.
The accounting should separate fixed overhead from optimizer-loop work and
reconcile per-run script timing against external subprocess wall time.

## Methods

- Method key: `direct_cp_8_4_2_be_bspline_analytic_multistage`
- Alpha: `1000`
- Stage cps: `8,4,2`
- Stage downsample: `4,2,1`
- Original MR-D-BSCP / DirectCP `8-4-2`:
  - lr `1e-3`
  - stage epochs `400,800,1300`
  - early stopping `none`
- `A_fast_decay`:
  - base lr `1e-2`
  - stage lr `3e-2,1e-2,3e-3`
  - max stage epochs `400,800,1300`
  - adaptive min epochs `75,150,250`
  - patience `50,75,100`
  - relative thresholds `1e-4` total loss and `5e-5` data loss

## Dataset And Split

- Dataset: OASIS BrainMR
- Split: seed-3 shuffled `pair_test_200`
- Rank-0 validation case: fixed `0333`, moving `0115`, original sorted index
  `158`
- First10: ranks `0-9` from the seed-3 shuffled protocol
- Foreground convention: raw fixed image-positive mask

## Execution

- Container: `py310sitreg_v1_test1`
- GPU policy: serial execution on GPU `0`; the launcher never schedules work
  on GPUs `1-3`
- Output root:
  `<external-run-root>/runs/oasis_directcp842_timing_breakdown_alpha1000_v1`
- Review root:
  `experiments/2026-08-04_oasis_directcp842_timing_breakdown`
- Timing mode: `--timing-detail deep`

## Commands

Tiny smoke:

```bash
python scripts/experiments/oasis_alpha_sweep/run_directcp842_timing_breakdown.py \
  --phase smoke --gpu 0 --overwrite
```

Rank-0 serial pair:

```bash
python scripts/experiments/oasis_alpha_sweep/run_directcp842_timing_breakdown.py \
  --phase rank0 --gpu 0 --overwrite
```

First10 serial pair, only after rank-0 accounting is acceptable:

```bash
python scripts/experiments/oasis_alpha_sweep/run_directcp842_timing_breakdown.py \
  --phase first10 --gpu 0 --overwrite
```

## Verification

- `py_compile`:
  - `scripts/experiments/oasis_alpha_sweep/run_one.py`
  - `scripts/experiments/oasis_alpha_sweep/run_directcp842_timing_breakdown.py`
  - `scripts/experiments/oasis_alpha_sweep/analyze_directcp842_timing_breakdown.py`
- Smoke: both candidate jobs write `DONE`, `summary.json`, and `timing.json`
- Rank-0: both full candidate jobs write `DONE`, `summary.json`, and
  `timing.json`
- Analyzer artifacts:
  - `timing_reconciliation.csv`
  - `timing_stage_breakdown.csv`
  - `timing_method_comparison.csv`
  - `timing_stage_linear_fits.csv`
  - `timing_report.md`
