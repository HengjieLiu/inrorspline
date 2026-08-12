# OASIS cps2 Runtime Sweep: D-BSCP And SINR

- Generated: `2026-08-07 05:20:57 +0000`
- Out root: `<container-data-root>/inrorspline/runs/oasis_cps2_single_stage_runtime_sweep_alpha1000_v1`
- Review root: `<repo-root>/experiments/2026-08-06_oasis_cps2_single_stage_runtime_sweep`
- Summaries: `120`
- Case ranks: `0-9`
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

- `D2_A_lr3e-2`: Dice `0.8004` (delta `-0.0020`), HD95 `2.4368` (delta `0.1580`), total `133.4s`, saving `36.9%`.
- `D2_A_lr1e-2`: Dice `0.8022` (delta `-9.087e-05`), HD95 `2.3277` (delta `0.0489`), total `145.6s`, saving `31.1%`.
- `D2_F400_lr3e-2`: Dice `0.7998` (delta `-0.0025`), HD95 `2.4437` (delta `0.1650`), total `150.8s`, saving `28.7%`.
- `D2_F600_lr1e-2`: Dice `0.8022` (delta `-1.816e-04`), HD95 `2.3347` (delta `0.0560`), total `157.7s`, saving `25.4%`.
- `S2_F1000_lr1e-3`: Dice `0.7918` (delta `-0.0074`), HD95 `2.2848` (delta `0.0474`), total `227.3s`, saving `43.5%`.
- `S2_A_lr1e-3`: Dice `0.7906` (delta `-0.0087`), HD95 `2.2449` (delta `0.0074`), total `241.5s`, saving `40.0%`.
- `S2_F1500_lr3e-4`: Dice `0.7948` (delta `-0.0045`), HD95 `2.1884` (delta `-0.0491`), total `285.1s`, saving `29.2%`.

## Candidate Table

| candidate | family | n | Dice | dDice | HD95 | dHD95 | total s | save % | epochs | cps4 ctx | promising |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2_baseline_lr1e-3_e2500 | D-BSCP | 10 | 0.8023 | 0.0000 | 2.2788 | 0.0000 | 211.4 | 0.0 | 2.5e+03 | yes | no |
| D2_A_lr3e-2 | D-BSCP | 10 | 0.8004 | -0.0020 | 2.4368 | 0.1580 | 133.4 | 36.9 | 214.3 | no | yes |
| D2_A_lr1e-2 | D-BSCP | 10 | 0.8022 | -9.087e-05 | 2.3277 | 0.0489 | 145.6 | 31.1 | 379.7 | no | yes |
| D2_F400_lr3e-2 | D-BSCP | 10 | 0.7998 | -0.0025 | 2.4437 | 0.1650 | 150.8 | 28.7 | 400.0 | no | yes |
| D2_F600_lr1e-2 | D-BSCP | 10 | 0.8022 | -1.816e-04 | 2.3347 | 0.0560 | 157.7 | 25.4 | 600.0 | no | yes |
| D2_F1000_lr3e-3 | D-BSCP | 10 | 0.8020 | -3.569e-04 | 2.2976 | 0.0188 | 165.9 | 21.5 | 1e+03 | yes | no |
| S2_baseline_lr1e-4_e2500 | SINR | 10 | 0.7992 | 0.0000 | 2.2374 | 0.0000 | 402.5 | 0.0 | 2.5e+03 | yes | no |
| S2_A_lr3e-3 | SINR | 10 | 0.6070 | -0.1922 | 3.6031 | 1.3657 | 180.0 | 55.3 | 495.2 | no | no |
| S2_F600_lr3e-3 | SINR | 10 | 0.6057 | -0.1935 | 3.6031 | 1.3656 | 184.5 | 54.2 | 600.0 | no | no |
| S2_F1000_lr1e-3 | SINR | 10 | 0.7918 | -0.0074 | 2.2848 | 0.0474 | 227.3 | 43.5 | 1e+03 | no | yes |
| S2_A_lr1e-3 | SINR | 10 | 0.7906 | -0.0087 | 2.2449 | 0.0074 | 241.5 | 40.0 | 1.04e+03 | no | yes |
| S2_F1500_lr3e-4 | SINR | 10 | 0.7948 | -0.0045 | 2.1884 | -0.0491 | 285.1 | 29.2 | 1.5e+03 | yes | yes |

## Files

- `candidate_summary.csv/json`: population means, deltas, gates, and cps4/MR-D-BSCP contextual comparisons.
- `case_results.csv`: per-case full-pipeline results.
- `accuracy_runtime_tradeoff.csv`: candidates sorted by gate status and runtime saving.
- `promising_candidates.csv`: candidates selected for sequential timing follow-up.
- `stage_summaries_combined.csv`: single-stage epoch and timing metadata.
- `job_manifest.csv/json`: launcher manifest.
- `launch_summary.json`: launcher completion summary.
