# Technical Log: DirectCP Production DVF Timing

## Known Limitations

- Rank-0 timing characterization only.
- DVF-only production mode intentionally excludes evaluation metrics and full
  packaging.
- CPU/GPU benchmark uses deterministic synthetic control grids with the same
  schedule shapes, not trained controls.

## Launch Notes

- Date: 2026-08-04.
- Container: `py310sitreg_v1_test1`.
- GPU policy: serial subprocesses on GPU `0`; GPUs `1-3` intentionally idle.
- Production output root:
  `<external-run-root>/runs/oasis_directcp842_production_dvf_timing_alpha1000_v1`.

## Results Log

- Production smoke completed for both methods with `1,1,1` epochs. Stage 1/2
  intermediate dense expansion was skipped; final dense DVF, `summary.json`,
  `timing.json`, and `DONE` were present.
- Diagnostic smoke with `--prolongation-preserve-check` completed for original
  MR-D-BSCP and split the old conservative transition path into pure knot
  insertion plus preservation dense expansion/metrics. In the diagnostic smoke,
  pure knot insertion was sub-millisecond, while preservation dense expansion
  was about `25.27 s` for `8->4` and `30.16 s` for `4->2`.
- CPU/GPU transition benchmark passed tolerance for all `7` rows
  (`max_abs_diff=0`, tolerance `1e-5`). GPU medians were faster for dense
  expansion (`33.50x`, `55.22x`, `78.05x` for cps `8,4,2`) and pure knot
  insertion (`2.26x` for `8->4`, `15.87x` for `4->2`). These operations are
  already PyTorch tensor operations, so normal GPU DirectCP runs already execute
  them on CUDA tensors.
- Full rank-0 production pair completed `2/2` jobs with no evaluation or
  full-output events counted in the DVF-only endpoint. Reconciliation residual
  failures were `0`; parent self-unaccounted warnings were `0`; non-target GPU
  compute-app observations were `0`.

## Rank-0 Production Timing

| Method | Epochs | Full script -> DVF memory | Loaded images -> DVF memory | Full script -> DVF file | External wall |
|---|---:|---:|---:|---:|---:|
| Original MR-D-BSCP / DirectCP `8-4-2` | 2500 | 174.72 s | 149.83 s | 174.78 s | 177.85 s |
| `A_fast_decay` | 600 | 129.29 s | 101.27 s | 129.36 s | 132.00 s |

Savings for `A_fast_decay` relative to original were `45.42 s` to in-memory
DVF, `45.42 s` to DVF file, and `45.84 s` command wall using the full-script
start. From loaded images, the savings were `48.56 s` to both memory and file.

## Conservative-To-Production Comparison

The previous DVF-only timing was conservative because it included intermediate
diagnostic dense expansion/preservation checking. Production timing excludes
that work unless explicitly requested.

| Method | Previous full script -> DVF memory | Production full script -> DVF memory | Saved |
|---|---:|---:|---:|
| Original MR-D-BSCP / DirectCP `8-4-2` | 234.18 s | 174.72 s | 59.46 s |
| `A_fast_decay` | 175.22 s | 129.29 s | 45.93 s |

The main avoidable overhead was diagnostic dense expansion/preservation
checking, not missing GPU acceleration. Pure knot insertion in the production
path is about `1 ms` total across transitions on the rank-0 case.

## Primary Artifacts

- `dvf_only_timing_report.md`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `dvf_only_grouped_breakdown.csv`
- `dvf_only_reconciliation.csv`
- `dvf_only_previous_conservative_comparison.csv`
- `command_timing_dvf_only_rank0.csv`
- `transition_kernel_benchmark.csv`
- `transition_kernel_benchmark_report.md`
