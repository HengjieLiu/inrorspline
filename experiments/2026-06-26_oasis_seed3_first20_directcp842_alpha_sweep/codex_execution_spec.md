# OASIS DirectCP 8-4-2 Remaining Alpha Sweep Execution Spec

## Goal

Complete the OASIS seed-3 first20 DirectCP `8,4,2` alpha sweep by launching
only the remaining five alphas and reusing the completed alpha `1000` root from
the preliminary run.

## Dataset And Split

- Dataset: OASIS BrainMR.
- Split: seed-3 shuffled ranks `0-19` from `pair_test_200`, resolved through
  the standard OASIS helper.
- Foreground convention: fixed raw image-positive mask (`fixed_image > 0`) for
  foreground regularity; segmentation masks are used only for label metrics.

## Method

- Method key: `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- Objective: full-image global NCC plus `alpha *` analytic cubic B-spline BE.
- Stage full cps: `8,4,2`.
- Stage downsample: `4,2,1`.
- Stage epochs: `400,800,1300`.
- Knot insertion: exact cubic B-spline dyadic prolongation between stages.
- Learning rate: `1e-2`.
- Seed: `1`.
- Eval interval: `250`.

## Alpha Handling

- New alphas to launch: `0,10,100,10000,100000`.
- Reused alpha: `1000`.
- Reused alpha1000 root:
  `/database/inrorspline/runs/oasis_seed3_first10_directcp842_alpha1000_prelim_v1`.
- The remaining-alpha launcher rejects alpha `1000`.

## Launch

- Remaining-alpha root:
  `/database/inrorspline/runs/oasis_seed3_first20_directcp842_alpha_sweep_remaining5_v1`.
- Launcher:
  `scripts/experiments/oasis_alpha_sweep/launch_directcp842_remaining_alpha_sweep.py`.
- GPUs: `0,1,2,3`.
- Max concurrency: `4`.
- Docker: `py310sitreg_v1_test1`, image `hengjieliu96/dl_dir:py310sitreg_v1`.

## Aggregation And Figures

- Aggregator:
  `scripts/experiments/oasis_alpha_sweep/aggregate_directcp842_full_alpha_sweep.py`.
- Required combined rows: `120` = ranks `0-19` x six alphas.
- Review root:
  `experiments/2026-06-26_oasis_seed3_first20_directcp842_alpha_sweep/`.
- Figure script:
  `scripts/experiments/oasis_alpha_sweep/plot_first10_first20_alpha5_saved_baselines_plus_directcp842.py`.
- Figure root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/pareto_first10_first20_alpha5_saved_baselines_plus_directcp842_paper_v1/`.
- Existing five-sweep figure directories must not be overwritten.

## Verification Plan

1. Compile changed scripts inside Docker.
2. Smoke rank `0`, alpha `0`, stage epochs `1,1,1`.
3. Dry-run launcher and verify exactly `100` jobs, ranks `0-19`, the five
   remaining alphas, GPUs `0,1,2,3`, and no alpha `1000`.
4. Launch remaining sweep and require `100/100` new `DONE` jobs.
5. Aggregate and require `100` remaining rows, `20` reused alpha1000 rows, and
   `120` combined DirectCP `8,4,2` rows.
6. Generate the plus-DirectCP842 figure package and verify first10/first20
   population row counts, sweep row counts, and non-empty PDF/PNG outputs.
