# OASIS DirectCP 8-4-2 Runtime Sweep

- Generated: `2026-08-04 09:22:33 +0000`
- Out root: `<external-run-root>/runs/oasis_directcp842_runtime_sweep_alpha1000_v1`
- Review root: `<repo-root>/experiments/2026-08-04_oasis_directcp842_runtime_sweep/first10`
- Summaries: `140`
- Case ranks: `0-9`
- Candidates: `F600_decay,F600_lr3e-2,A_bal_lr3e-2,A_bal_lr1e-2,A_fast_decay,F600_lr1e-2,F950_decay,F1550_lr3e-3,F950_lr1e-2,A_safe_lr1e-2,F950_lr3e-2,A_safe_lr3e-2,F1550_lr3e-2,F1550_lr1e-2`
- Alpha: `1000`
- Variant: `oasis_seed3_first10_directcp842_runtime_sweep_alpha1000_v1`
- Baseline source: `<repo-root>/experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization/summary_runs_combined.csv`
- Matched baseline Dice/HD95: `0.8070` / `2.2020`

## Acceptance

- PASS: all expected runtime-sweep rows and artifacts are present.
- Fastest smoke-passing candidate: `A_fast_decay` with total `183.2s`, Dice `0.8058`, HD95 `2.2147`.
- Smoke-promoted candidates: `A_fast_decay,F600_decay,F600_lr1e-2,F600_lr3e-2,A_bal_lr1e-2,A_bal_lr3e-2,F950_lr1e-2,F950_lr3e-2,A_safe_lr3e-2,F950_decay,A_safe_lr1e-2,F1550_lr3e-3,F1550_lr3e-2,F1550_lr1e-2`

## Candidate Table

| candidate | tier | n | Dice | HD95 | train s | total s | epochs | smoke | runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_fast_decay | A | 10 | 0.8058 | 2.2147 | 149.5 | 183.2 | 577.5 | yes | no |
| F600_decay | A | 10 | 0.8063 | 2.2089 | 151.8 | 186.2 | 600.0 | yes | no |
| F600_lr1e-2 | A | 10 | 0.8059 | 2.2312 | 152.7 | 186.3 | 600.0 | yes | no |
| F600_lr3e-2 | A | 10 | 0.8041 | 2.2691 | 154.0 | 188.7 | 600.0 | yes | no |
| A_bal_lr1e-2 | A | 10 | 0.8061 | 2.2147 | 154.7 | 190.0 | 720.0 | yes | no |
| A_bal_lr3e-2 | A | 10 | 0.8046 | 2.2641 | 156.8 | 190.6 | 702.5 | yes | no |
| F950_lr1e-2 | A | 10 | 0.8061 | 2.2244 | 157.9 | 193.3 | 950.0 | yes | no |
| F950_lr3e-2 | A | 10 | 0.8047 | 2.2572 | 157.9 | 193.7 | 950.0 | yes | no |
| A_safe_lr3e-2 | A | 10 | 0.8036 | 2.2567 | 160.6 | 194.3 | 1.06e+03 | yes | no |
| F950_decay | A | 10 | 0.8057 | 2.2186 | 160.7 | 194.6 | 950.0 | yes | no |
| A_safe_lr1e-2 | A | 10 | 0.8054 | 2.2445 | 163.2 | 198.6 | 1.14e+03 | yes | no |
| F1550_lr3e-3 | A | 10 | 0.8071 | 2.2124 | 172.1 | 205.8 | 1.55e+03 | yes | no |
| F1550_lr3e-2 | A | 10 | 0.8039 | 2.2644 | 176.1 | 208.8 | 1.55e+03 | yes | no |
| F1550_lr1e-2 | A | 10 | 0.8056 | 2.2329 | 180.3 | 214.5 | 1.55e+03 | yes | no |

## Files

- `summary_runs.csv/json`: per-case candidate summaries.
- `candidate_population_summary.csv/json`: candidate population means and spreads.
- `candidate_acceptance.csv/json`: smoke, tier, and runtime target decisions.
- `promoted_candidates.txt`: comma-separated smoke-passing candidates.
- `stage_summaries_combined.csv`: compact per-stage timing and stop reasons.
