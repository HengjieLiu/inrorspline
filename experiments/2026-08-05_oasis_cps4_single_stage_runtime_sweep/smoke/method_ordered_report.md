# OASIS cps4 Runtime Sweep: D-BSCP And SINR

- Generated: `2026-08-05 23:28:17 +0000`
- Out root: `<external-run-root>/runs/oasis_cps4_single_stage_runtime_sweep_alpha1000_v1_smoke`
- Review root: `experiments/2026-08-05_oasis_cps4_single_stage_runtime_sweep/smoke`
- Summaries: `12`
- Case ranks: `0`
- Candidates: `all`
- Alpha: `1000`
- Variant: `oasis_seed3_first10_cps4_single_stage_runtime_sweep_alpha1000_v1`
- Gate: Dice drop `<= 0.02` and HD95 increase `<= 0.2 mm` versus same-method cps4 baseline.
- Runtime target: full-pipeline total elapsed improvement `>= 25%`, or `>= 25%` epoch reduction with no Dice/HD95 loss.
- cps2 anchors: DirectCP cps2 Dice/HD95 `0.8023/2.2788`; SINR cps2 `0.7992/2.2374`.

## Validation

- PASS: all expected rows and artifacts are present.

## Promising Candidates

- None under the current gate.

## Candidate Table

| candidate | family | n | Dice | dDice | HD95 | dHD95 | total s | save % | epochs | cps2 | promising |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_baseline_lr1e-3_e2500 | D-BSCP | 1 | 0.5332 | 0.0000 | 4.5466 | 0.0000 | 82.1 | 0.0 | 1.0 | no | no |
| D_A_lr3e-2 | D-BSCP | 1 | 0.5236 | -0.0096 | 4.4236 | -0.1230 | 72.5 | 11.7 | 1.0 | no | no |
| D_F600_lr1e-2 | D-BSCP | 1 | 0.5747 | 0.0415 | 4.1113 | -0.4353 | 79.1 | 3.7 | 1.0 | no | no |
| D_F1000_lr3e-3 | D-BSCP | 1 | 0.5332 | 0.0000 | 4.5466 | 0.0000 | 81.3 | 1.0 | 1.0 | no | no |
| D_F400_lr3e-2 | D-BSCP | 1 | 0.5236 | -0.0096 | 4.4236 | -0.1230 | 82.5 | -0.4 | 1.0 | no | no |
| D_A_lr1e-2 | D-BSCP | 1 | 0.5747 | 0.0415 | 4.1113 | -0.4353 | 84.0 | -2.3 | 1.0 | no | no |
| S_baseline_lr1e-4_e2500 | SINR | 1 | 0.4279 | 0.0000 | 6.2243 | 0.0000 | 80.2 | 0.0 | 1.0 | no | no |
| S_A_lr1e-3 | SINR | 1 | 0.0667 | -0.3613 | 24.3855 | 18.1612 | 76.4 | 4.7 | 1.0 | no | no |
| S_F1000_lr1e-3 | SINR | 1 | 0.0667 | -0.3613 | 24.3855 | 18.1612 | 77.3 | 3.6 | 1.0 | no | no |
| S_A_lr3e-3 | SINR | 1 | 0.2316 | -0.1963 | 16.7840 | 10.5597 | 79.3 | 1.1 | 1.0 | no | no |
| S_F1500_lr3e-4 | SINR | 1 | 0.2149 | -0.2130 | 12.6905 | 6.4662 | 84.6 | -5.4 | 1.0 | no | no |
| S_F600_lr3e-3 | SINR | 1 | 0.2316 | -0.1963 | 16.7840 | 10.5597 | 86.3 | -7.6 | 1.0 | no | no |

## Files

- `candidate_summary.csv/json`: population means, deltas, gates, and cps2 reference comparisons.
- `case_results.csv`: per-case full-pipeline results.
- `accuracy_runtime_tradeoff.csv`: candidates sorted by gate status and runtime saving.
- `promising_candidates.csv`: candidates selected for sequential timing follow-up.
- `stage_summaries_combined.csv`: single-stage epoch and timing metadata.
- `job_manifest.csv/json`: launcher manifest.
- `launch_summary.json`: launcher completion summary.
