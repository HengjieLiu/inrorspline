# Codex Execution Spec: OASIS DirectCP 8-4-2 DVF-Only Timing

## Objective

Measure controlled rank-0 runtime from OASIS input images to a dense DirectCP
`8-4-2` DVF, comparing original MR-D-BSCP / DirectCP `8-4-2` with
`A_fast_decay`.

Primary endpoint:

- dense DVF ready in memory, immediately after final-stage dense displacement
  expansion.

Secondary endpoint:

- dense DVF file ready on disk, immediately after normalized-grid dense
  displacement `.npy` write.

## Dataset And Case

- Dataset: OASIS BrainMR seed-3 shuffled `pair_test_200`.
- Rank: `0`.
- Fixed/moving: `(0333, 0115)`.
- Original sorted index: `158`.
- Pair file:
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.

## Methods

- `original_mrdbscp`: alpha `1000`, lr `1e-3`, cps `8,4,2`, downsample
  `4,2,1`, stage epochs `400,800,1300`, no early stop.
- `A_fast_decay`: alpha `1000`, base lr `1e-2`, stage lr
  `3e-2,1e-2,3e-3`, cps `8,4,2`, downsample `4,2,1`, max stage epochs
  `400,800,1300`, adaptive minimums `75,150,250`, patience `50,75,100`,
  relative thresholds `1e-4` total and `5e-5` data.

## Timing Mode

Run `run_one.py` with:

```text
--output-mode dvf_only
--timing-detail deep
```

`dvf_only` skips post-training evaluation, regularity metrics, model/control
checkpoint saves, and full CSV packaging. It writes only the dense DVF `.npy`,
minimal `summary.json`, `DONE`, and `timing.json`.

## Execution Order

1. `1,1,1` epoch smoke for both methods.
2. Analyze smoke timing and verify no evaluation/full-output events.
3. Full rank-0 original serially on GPU `0`.
4. Full rank-0 `A_fast_decay` serially on GPU `0`.
5. Generate final DVF-only timing report.

## Roots

- Output root:
  `<external-run-root>/runs/oasis_directcp842_dvf_only_timing_alpha1000_v1`
- Review root:
  `experiments/2026-08-04_oasis_directcp842_dvf_only_timing/`

## Review Artifacts

- `dvf_only_timing_report.md`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `dvf_only_grouped_breakdown.csv`
- `dvf_only_reconciliation.csv`
- `command_timing_dvf_only_rank0.csv`
