# OASIS DirectCP 8-4-2 Runtime Sweep

- Generated: `2026-08-04 07:50:42 +0000`
- Out root: `<external-run-root>/runs/oasis_directcp842_runtime_sweep_alpha1000_v1`
- Review root: `<repo-root>/experiments/2026-08-04_oasis_directcp842_runtime_sweep/smoke`
- Summaries: `48`
- Case ranks: `0,4,8`
- Candidates: `all`
- Alpha: `1000`
- Variant: `oasis_seed3_first10_directcp842_runtime_sweep_alpha1000_v1`
- Baseline source: `<repo-root>/experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization/summary_runs_combined.csv`
- Matched baseline Dice/HD95: `0.7708` / `2.6728`

## Acceptance

- PASS: all expected runtime-sweep rows and artifacts are present.
- Fastest smoke-passing candidate: `F600_decay` with total `176.1s`, Dice `0.7701`, HD95 `2.6675`.
- Smoke-promoted candidates: `F600_decay,F600_lr3e-2,A_bal_lr3e-2,A_bal_lr1e-2,A_fast_decay,F600_lr1e-2,F950_decay,F1550_lr3e-3,F950_lr1e-2,A_safe_lr1e-2,F950_lr3e-2,A_safe_lr3e-2,F1550_lr3e-2,F1550_lr1e-2`

## Candidate Table

| candidate | tier | n | Dice | HD95 | train s | total s | epochs | smoke | runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F600_decay | drop | 3 | 0.7701 | 2.6675 | 141.4 | 176.1 | 600.0 | yes | yes |
| F600_lr3e-2 | drop | 3 | 0.7646 | 2.7030 | 147.3 | 183.0 | 600.0 | yes | no |
| A_bal_lr3e-2 | drop | 3 | 0.7669 | 2.7101 | 152.5 | 187.3 | 708.3 | yes | no |
| A_bal_lr1e-2 | drop | 3 | 0.7694 | 2.6491 | 151.3 | 187.6 | 750.0 | yes | no |
| A_fast_decay | drop | 3 | 0.7696 | 2.6591 | 153.9 | 188.2 | 583.3 | yes | no |
| F600_lr1e-2 | drop | 3 | 0.7699 | 2.6823 | 158.7 | 189.7 | 600.0 | yes | no |
| F950_decay | drop | 3 | 0.7697 | 2.6527 | 160.1 | 192.6 | 950.0 | yes | no |
| F1550_lr3e-3 | drop | 3 | 0.7713 | 2.6567 | 159.4 | 193.2 | 1.55e+03 | yes | no |
| F950_lr1e-2 | drop | 3 | 0.7696 | 2.6686 | 161.3 | 193.4 | 950.0 | yes | no |
| A_safe_lr1e-2 | drop | 3 | 0.7694 | 2.6641 | 162.6 | 194.1 | 1.12e+03 | yes | no |
| F950_lr3e-2 | drop | 3 | 0.7657 | 2.7265 | 154.2 | 194.1 | 950.0 | yes | no |
| A_safe_lr3e-2 | drop | 3 | 0.7641 | 2.7093 | 166.7 | 200.3 | 1.04e+03 | yes | no |
| F1550_lr3e-2 | drop | 3 | 0.7655 | 2.7025 | 174.1 | 206.1 | 1.55e+03 | yes | no |
| F1550_lr1e-2 | drop | 3 | 0.7689 | 2.6727 | 179.8 | 213.9 | 1.55e+03 | yes | no |
| F600_lr1e-1 | drop | 3 | 0.7071 | 4.4376 | 159.5 | 195.0 | 600.0 | no | no |
| F950_lr1e-1 | drop | 3 | 0.6995 | 5.0727 | 162.9 | 199.7 | 950.0 | no | no |

## Files

- `summary_runs.csv/json`: per-case candidate summaries.
- `candidate_population_summary.csv/json`: candidate population means and spreads.
- `candidate_acceptance.csv/json`: smoke, tier, and runtime target decisions.
- `promoted_candidates.txt`: comma-separated smoke-passing candidates.
- `stage_summaries_combined.csv`: compact per-stage timing and stop reasons.
