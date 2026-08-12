# OASIS cps2 Runtime Sweep: D-BSCP And SINR

- Generated: `2026-08-07 03:23:51 +0000`
- Out root: `<container-data-root>/inrorspline/runs/oasis_cps2_single_stage_runtime_sweep_alpha1000_v1_smoke`
- Review root: `<repo-root>/experiments/2026-08-06_oasis_cps2_single_stage_runtime_sweep/smoke`
- Summaries: `12`
- Case ranks: `0`
- Candidates: `all`
- Alpha: `1000`
- Variant: `oasis_seed3_first10_cps2_single_stage_runtime_sweep_alpha1000_v1`
- Gate: Dice drop `<= 0.02` and HD95 increase `<= 0.2 mm` versus same-method cps2 baseline.
- Runtime target: full-pipeline total elapsed improvement `>= 25%`, or `>= 25%` epoch reduction with no Dice/HD95 loss.
- cps4 context anchors: D-BSCP cps4 baseline Dice/HD95 `0.7965/2.3120`; SINR cps4 baseline `0.7945/2.2745`.
- MR-D-BSCP context anchors: original Dice/HD95 `0.8070/2.2020`; A_fast_decay `0.8058/2.2147`.

## Validation

- PASS: all expected rows and artifacts are present.

## Promising Candidates

- None under the current gate.

## Candidate Table

| candidate | family | n | Dice | dDice | HD95 | dHD95 | total s | save % | epochs | cps4 ctx | promising |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2_baseline_lr1e-3_e2500 | D-BSCP | 1 | 0.5332 | 0.0000 | 4.5466 | 0.0000 | 119.1 | 0.0 | 1.0 | no | no |
| D2_A_lr1e-2 | D-BSCP | 1 | 0.5858 | 0.0525 | 4.0429 | -0.5037 | 117.5 | 1.3 | 1.0 | no | no |
| D2_F400_lr3e-2 | D-BSCP | 1 | 0.5152 | -0.0180 | 4.4277 | -0.1189 | 117.6 | 1.3 | 1.0 | no | no |
| D2_F600_lr1e-2 | D-BSCP | 1 | 0.5858 | 0.0525 | 4.0429 | -0.5037 | 124.1 | -4.2 | 1.0 | no | no |
| D2_A_lr3e-2 | D-BSCP | 1 | 0.5152 | -0.0180 | 4.4277 | -0.1189 | 125.7 | -5.5 | 1.0 | no | no |
| D2_F1000_lr3e-3 | D-BSCP | 1 | 0.5332 | 0.0000 | 4.5466 | 0.0000 | 133.6 | -12.2 | 1.0 | no | no |
| S2_baseline_lr1e-4_e2500 | SINR | 1 | 0.4283 | 0.0000 | 6.2690 | 0.0000 | 114.8 | 0.0 | 1.0 | no | no |
| S2_A_lr3e-3 | SINR | 1 | 0.1747 | -0.2536 | 21.9839 | 15.7150 | 131.1 | -14.2 | 1.0 | no | no |
| S2_F1500_lr3e-4 | SINR | 1 | 0.2153 | -0.2130 | 12.8001 | 6.5311 | 131.7 | -14.7 | 1.0 | no | no |
| S2_F1000_lr1e-3 | SINR | 1 | 0.0675 | -0.3607 | 24.0024 | 17.7335 | 132.1 | -15.0 | 1.0 | no | no |
| S2_A_lr1e-3 | SINR | 1 | 0.0675 | -0.3607 | 24.0024 | 17.7335 | 132.7 | -15.5 | 1.0 | no | no |
| S2_F600_lr3e-3 | SINR | 1 | 0.1747 | -0.2536 | 21.9839 | 15.7150 | 132.8 | -15.6 | 1.0 | no | no |

## Files

- `candidate_summary.csv/json`: population means, deltas, gates, and cps4/MR-D-BSCP contextual comparisons.
- `case_results.csv`: per-case full-pipeline results.
- `accuracy_runtime_tradeoff.csv`: candidates sorted by gate status and runtime saving.
- `promising_candidates.csv`: candidates selected for sequential timing follow-up.
- `stage_summaries_combined.csv`: single-stage epoch and timing metadata.
- `job_manifest.csv/json`: launcher manifest.
- `launch_summary.json`: launcher completion summary.
