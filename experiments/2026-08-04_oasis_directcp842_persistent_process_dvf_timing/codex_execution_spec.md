# Codex Execution Spec: DirectCP Persistent-Process DVF Timing

## Goal

Measure DirectCP `8-4-2` DVF-only runtime for OASIS seed-3 shuffled ranks
`0-9` in a persistent-process setting.

## Process Model

- Use one long-lived Python worker subprocess per candidate.
- Run ranks `0-9` sequentially inside that worker.
- Treat rank `0` as the cold first case.
- Treat ranks `1-9` as warm persistent-process cases.
- Run original and `A_fast_decay` in separate worker subprocesses so the second
  method does not inherit the first method's CUDA/cuDNN warm state.

## Methods

- Original MR-D-BSCP / DirectCP `8-4-2`: alpha `1000`, lr `1e-3`, cps
  `8,4,2`, downsample `4,2,1`, stage epochs `400,800,1300`, no early stop.
- `A_fast_decay`: alpha `1000`, base lr `1e-2`, stage lr
  `3e-2,1e-2,3e-3`, cps `8,4,2`, downsample `4,2,1`, max stage epochs
  `400,800,1300`, adaptive minimums `75,150,250`, patience `50,75,100`.

## Runtime Endpoint

Primary endpoint is in-memory dense DVF. Also report dense DVF file-ready time.
Production DVF-only mode excludes evaluation, regularity metrics, checkpoint
saves, and full packaging.

## Outputs

- `persistent_process_timing_report.md`
- `persistent_process_summary.csv`
- `persistent_process_case_timing.csv`
- `persistent_process_savings.csv`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `command_timing_dvf_only_first10.csv`

## Acceptance

- Smoke run completes for both methods.
- Full first10 completes `20/20` case rows.
- Dense DVF, `summary.json`, `timing.json`, and `DONE` exist for each case.
- Evaluation/full-output events are `0`.
- Timing reconciliation failures are `0`.
