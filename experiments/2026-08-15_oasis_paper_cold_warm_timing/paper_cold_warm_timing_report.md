# OASIS Paper Cold/Warm DVF Timing

- Variant: `oasis_paper_cold_warm_timing_alpha1000_v1`
- Generated: `2026-08-15 09:48:38 +0000`
- Completed valid timing rows: `110/110`
- Missing or failed rows: `0`

## Timing Definitions

- `cold`: first OASIS rank-0 run in a fresh method worker.
- `warm`: rank 0 rerun plus ranks 1-9 in the same already-initialized worker.
- `loaded_images_to_dvf_file_sec`: paper endpoint, loaded input images to saved dense DVF.
- `full_script_to_dvf_file_sec`: script start to saved dense DVF.
- `persistent_case_wall_sec`: worker-measured wall around one in-process `run_one.main()` call.
- All runs use `--output-mode dvf_only`, so Dice/HD95/ASD/dense NCC/regularity evaluation is skipped.
- Step counts are optimizer updates for one image pair; the report labels them as steps, not epochs.

## Method Summary
| Phase | Candidate | Method | Cold n | Warm n | Cold DVF s | Warm DVF s | Warm SD | Warm steps | Peak GB |
|---|---|---|---|---|---|---|---|---|---|
| fast | MR_A_fast_decay | MR-D-BSCP (8-4-2) | 1 | 10 | 97.16 | 10.08 | 0.87 | 437.50 | 3.86 |
| fast | D2_A_bal_lr10x | D-BSCP (cps2) | 1 | 10 | 63.23 | 12.89 | 1.26 | 400.00 | 2.09 |
| fast | D4_A_bal_lr10x | D-BSCP (cps4) | 1 | 10 | 65.94 | 15.50 | 1.68 | 476.90 | 1.75 |
| fast | S2_A_bal_lr10x | INR-BSCP (cps2) | 1 | 10 | 163.21 | 90.20 | 11.01 | 820.80 | 7.63 |
| fast | S4_A_bal_lr10x | INR-BSCP (cps4) | 1 | 10 | 77.89 | 28.93 | 5.25 | 671.20 | 2.50 |
| baseline | MR_baseline_lr1x_full | MR-D-BSCP (8-4-2) | 1 | 10 | 155.20 | 61.31 | 0.82 | 2500.00 | 3.86 |
| baseline | D2_baseline_lr1x_full | D-BSCP (cps2) | 1 | 10 | 131.41 | 80.67 | 0.23 | 2500.00 | 2.09 |
| baseline | D4_baseline_lr1x_full | D-BSCP (cps4) | 1 | 10 | 125.04 | 81.33 | 0.23 | 2500.00 | 1.75 |
| baseline | S2_baseline_lr1x_full | INR-BSCP (cps2) | 1 | 10 | 336.98 | 275.22 | 3.14 | 2500.00 | 7.63 |
| baseline | S4_baseline_lr1x_full | INR-BSCP (cps4) | 1 | 10 | 167.42 | 108.12 | 0.10 | 2500.00 | 2.50 |

## Fast Versus 2500-Step Baseline
| Method | Fast | Baseline | Cold base | Cold fast | Cold saving % | Warm base | Warm fast | Warm saving % |
|---|---|---|---|---|---|---|---|---|
| MR-D-BSCP (8-4-2) | MR_A_fast_decay | MR_baseline_lr1x_full | 155.20 | 97.16 | 37.40 | 61.31 | 10.08 | 83.56 |
| D-BSCP (cps2) | D2_A_bal_lr10x | D2_baseline_lr1x_full | 131.41 | 63.23 | 51.88 | 80.67 | 12.89 | 84.02 |
| D-BSCP (cps4) | D4_A_bal_lr10x | D4_baseline_lr1x_full | 125.04 | 65.94 | 47.27 | 81.33 | 15.50 | 80.95 |
| INR-BSCP (cps2) | S2_A_bal_lr10x | S2_baseline_lr1x_full | 336.98 | 163.21 | 51.57 | 275.22 | 90.20 | 67.23 |
| INR-BSCP (cps4) | S4_A_bal_lr10x | S4_baseline_lr1x_full | 167.42 | 77.89 | 53.47 | 108.12 | 28.93 | 73.24 |

## Paper Runtime Table Values
| Method | Dice | HD95 | Steps | Cold s | Warm s | Peak GB |
|---|---|---|---|---|---|---|
| MR-D-BSCP (8-4-2) | 0.810->0.809 | 2.17->2.18 | 2500->438 | 155.2->97.2 | 61.3->10.1 | 3.86->3.86 |
| D-BSCP (cps2) | 0.805->0.805 | 2.33->2.33 | 2500->400 | 131.4->63.2 | 80.7->12.9 | 2.09->2.09 |
| D-BSCP (cps4) | 0.799->0.797 | 2.29->2.37 | 2500->477 | 125.0->65.9 | 81.3->15.5 | 1.75->1.75 |
| INR-BSCP (cps2) | 0.799->0.793 | 2.31->2.24 | 2500->821 | 337.0->163.2 | 275.2->90.2 | 7.63->7.63 |
| INR-BSCP (cps4) | 0.796->0.786 | 2.38->2.25 | 2500->671 | 167.4->77.9 | 108.1->28.9 | 2.50->2.50 |

## Outputs

- `paper_cold_warm_case_timing.csv/json`
- `paper_cold_warm_method_summary.csv/json`
- `paper_fast_vs_baseline_timing.csv`
- `paper_runtime_table_values.csv`
- `missing_or_failed_jobs.json`
- `gpu_observations.csv`
