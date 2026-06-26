# OASIS DirectCP 8-4-2 Alpha1000 Preliminary Execution Spec

## Objective

Run a seed-3 first10 preliminary check of DirectCP multistage `8 -> 4 -> 2`
on OASIS BrainMR at alpha `1000`, then compare it against existing first10
IDIR/SINR/DirectCP operating points.

## Dataset

- Dataset: OASIS BrainMR `pair_test_200`.
- Pair order: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case ranks: `0-9`.
- Original sorted indices: `158,62,14,69,47,137,110,6,116,88`.
- Foreground metric mask: fixed raw image-positive, `fixed_image > 0`.
- Segmentation masks: Dice/HD95/ASD only.

## Method

- Method key: `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- Stage cps: `8,4,2`.
- Stage downsample: `4,2,1`.
- Stage epochs: `400,800,1300`.
- Learning rate: `1e-2`.
- Alpha: `1000`.
- Seed: `1`.
- Optimizer: Adam.
- Prolongation: exact cubic B-spline dyadic knot insertion.
- Loss: full-image global NCC plus analytic cubic B-spline BE.

## Paths

- Large output root:
  `/database/inrorspline/runs/oasis_seed3_first10_directcp842_alpha1000_prelim_v1`
- Review root:
  `experiments/2026-06-26_oasis_seed3_first10_directcp842_alpha1000_prelim/`
- Reference comparison root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_full_overlay/`

## Launch

Use Docker container `py310sitreg_v1_test1` and GPUs `2,3` only:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_directcp842_alpha1000_prelim.py --dry-run
python scripts/experiments/oasis_alpha_sweep/launch_directcp842_alpha1000_prelim.py
```

## Acceptance

- Compile/import passes for changed scripts.
- Rank-0 smoke with stage epochs `1,1,1` passes.
- Dry-run queues exactly `10` jobs on GPUs `2,3`.
- Full run completes `10/10` summaries.
- Aggregation writes `comparison_population_comprehensive.csv/json`,
  `comparison_population_main.csv`, and `comparison_report.md`.
