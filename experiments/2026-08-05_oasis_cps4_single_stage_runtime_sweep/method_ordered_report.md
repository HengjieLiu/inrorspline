# OASIS cps4 Runtime Sweep: D-BSCP And SINR

- Generated: `2026-08-06 00:42:31 +0000`
- Out root: `<external-run-root>/runs/oasis_cps4_single_stage_runtime_sweep_alpha1000_v1`
- Review root: `<repo-root>/experiments/2026-08-05_oasis_cps4_single_stage_runtime_sweep`
- Summaries: `120`
- Case ranks: `0-9`
- Candidates: `all`
- Alpha: `1000`
- Variant: `oasis_seed3_first10_cps4_single_stage_runtime_sweep_alpha1000_v1`
- Gate: Dice drop `<= 0.02` and HD95 increase `<= 0.2 mm` versus same-method cps4 baseline.
- Runtime target: full-pipeline total elapsed improvement `>= 25%`, or `>= 25%` epoch reduction with no Dice/HD95 loss.
- cps2 anchors: DirectCP cps2 Dice/HD95 `0.8023/2.2788`; SINR cps2 `0.7992/2.2374`.

## Validation

- PASS: all expected rows and artifacts are present.

## Promising Candidates

- `D_A_lr1e-2`: Dice `0.7946` (delta `-0.0019`), HD95 `2.3653` (delta `0.0533`), total `105.4s`, saving `35.9%`.
- `D_F600_lr1e-2`: Dice `0.7949` (delta `-0.0016`), HD95 `2.3795` (delta `0.0676`), total `117.7s`, saving `28.4%`.
- `D_F1000_lr3e-3`: Dice `0.7958` (delta `-6.636e-04`), HD95 `2.3343` (delta `0.0224`), total `120.0s`, saving `27.0%`.
- `S_F1000_lr1e-3`: Dice `0.7866` (delta `-0.0079`), HD95 `2.2503` (delta `-0.0242`), total `131.0s`, saving `31.7%`.
- `S_A_lr1e-3`: Dice `0.7813` (delta `-0.0132`), HD95 `2.2331` (delta `-0.0414`), total `134.4s`, saving `29.9%`.

## Candidate Table

| candidate | family | n | Dice | dDice | HD95 | dHD95 | total s | save % | epochs | cps2 | promising |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_baseline_lr1e-3_e2500 | D-BSCP | 10 | 0.7965 | 0.0000 | 2.3120 | 0.0000 | 164.4 | 0.0 | 2.5e+03 | no | no |
| D_A_lr3e-2 | D-BSCP | 10 | 0.7889 | -0.0076 | 2.6665 | 0.3545 | 94.2 | 42.7 | 326.3 | no | no |
| D_F400_lr3e-2 | D-BSCP | 10 | 0.7884 | -0.0081 | 2.6447 | 0.3328 | 100.0 | 39.1 | 400.0 | no | no |
| D_A_lr1e-2 | D-BSCP | 10 | 0.7946 | -0.0019 | 2.3653 | 0.0533 | 105.4 | 35.9 | 476.9 | no | yes |
| D_F600_lr1e-2 | D-BSCP | 10 | 0.7949 | -0.0016 | 2.3795 | 0.0676 | 117.7 | 28.4 | 600.0 | no | yes |
| D_F1000_lr3e-3 | D-BSCP | 10 | 0.7958 | -6.636e-04 | 2.3343 | 0.0224 | 120.0 | 27.0 | 1e+03 | no | yes |
| S_baseline_lr1e-4_e2500 | SINR | 10 | 0.7945 | 0.0000 | 2.2745 | 0.0000 | 191.9 | 0.0 | 2.5e+03 | no | no |
| S_A_lr3e-3 | SINR | 10 | 0.5660 | -0.2285 | 3.8791 | 1.6046 | 109.0 | 43.2 | 350.0 | no | no |
| S_F600_lr3e-3 | SINR | 10 | 0.5640 | -0.2305 | 3.8803 | 1.6058 | 121.3 | 36.8 | 600.0 | no | no |
| S_F1000_lr1e-3 | SINR | 10 | 0.7866 | -0.0079 | 2.2503 | -0.0242 | 131.0 | 31.7 | 1e+03 | no | yes |
| S_A_lr1e-3 | SINR | 10 | 0.7813 | -0.0132 | 2.2331 | -0.0414 | 134.4 | 29.9 | 846.4 | no | yes |
| S_F1500_lr3e-4 | SINR | 10 | 0.7943 | -2.482e-04 | 2.1781 | -0.0963 | 148.4 | 22.7 | 1.5e+03 | no | no |

## Files

- `candidate_summary.csv/json`: population means, deltas, gates, and cps2 reference comparisons.
- `case_results.csv`: per-case full-pipeline results.
- `accuracy_runtime_tradeoff.csv`: candidates sorted by gate status and runtime saving.
- `promising_candidates.csv`: candidates selected for sequential timing follow-up.
- `stage_summaries_combined.csv`: single-stage epoch and timing metadata.
- `job_manifest.csv/json`: launcher manifest.
- `launch_summary.json`: launcher completion summary.
