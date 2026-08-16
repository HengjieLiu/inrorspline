---
doc_type: codex_execution_spec
status: active
---

# OASIS Paper Cold/Warm DVF Timing

## Purpose

Refresh the paper-facing OASIS runtime table for the selected accelerated
schedules and their 2500-step baselines. The timing endpoint is loaded input
images to saved dense DVF, including pair-specific optimization and DVF output,
excluding evaluation metrics.

## Methods

Fast schedules:

- `MR_A_fast_decay`
- `D2_A_bal_lr10x`
- `D4_A_bal_lr10x`
- `S2_A_bal_lr10x`
- `S4_A_bal_lr10x`

2500-step baselines:

- `MR_baseline_lr1x_full`
- `D2_baseline_lr1x_full`
- `D4_baseline_lr1x_full`
- `S2_baseline_lr1x_full`
- `S4_baseline_lr1x_full`

All methods use OASIS alpha `1000`, seed `1`, and the same candidate
definitions as the unified acceleration sweep.

## Timing Protocol

For each method:

1. Run OASIS rank `0` once in a fresh persistent worker and label it `cold`.
2. In the same worker, rerun rank `0`, then run ranks `1-9`; label these 10
   runs `warm`.
3. Record `loaded_images_to_dvf_file_sec`, `full_script_to_dvf_file_sec`,
   worker wall time, actual optimizer steps, and peak GPU memory.

The run uses `--output-mode dvf_only`; Dice, HD95, ASD, dense NCC, and
regularity evaluation are not performed in these timing jobs.

## Outputs

- Run root:
  `/database/inrorspline/runs/oasis_paper_cold_warm_timing_alpha1000_v1`
- Review root:
  `experiments/2026-08-15_oasis_paper_cold_warm_timing/`
- Live monitor:
  `live_monitor.md` and `live_monitor.csv`
- Final report:
  `paper_cold_warm_timing_report.md`

## Acceptance

- Dry-run manifest has `10` candidates and `110` measured registrations.
- Smoke has `20/20` completed one-step DVF-only runs.
- Full run has `110/110` completed DVF-only timing runs.
- All completed rows have finite input-to-DVF-file timing, existing dense DVF
  files, `output_mode=dvf_only`, and `no_evaluation=true`.
