# OASIS Seed-3 First10 Alpha Sweep Execution Spec

## Status

Prepared for review. Do not launch until explicitly approved.

## Objective

Run the OASIS seed-3 shuffled first10 BE alpha sweep and aggregate population
ARC-style curves. Population points are method/alpha means across the 10 cases;
case-level and per-label metrics remain available in CSV.

## Dataset

- Dataset: OASIS BrainMR `pair_test_200`.
- Pair order: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case ranks: `0-9`.
- Original sorted indices: `158,62,14,69,47,137,110,6,116,88`.
- Fixed/moving pairs:
  - `(0333, 0115)`
  - `(0113, 0017)`
  - `(0037, 0394)`
  - `(0117, 0322)`
  - `(0084, 0108)`
  - `(0288, 0024)`
  - `(0194, 0184)`
  - `(0014, 0149)`
  - `(0215, 0326)`
  - `(0173, 0136)`

## Matrix

- IDIR: lr `1e-4`, coordinate-autograd analytic BE, alpha
  `0,0.01,0.1,1,10,100`.
- SINR cps4: lr `1e-4`, cubic B-spline analytic BE, alpha
  `0,10,100,1000,10000,100000`.
- SINR cps2: lr `1e-4`, cubic B-spline analytic BE, alpha
  `0,10,100,1000,10000,100000`.
- DirectCP cps4: lr `1e-3`, cubic B-spline analytic BE, alpha
  `0,10,100,1000,10000,100000`.
- Epochs: `2500`.
- Eval interval: `250`.
- Seed: `1`.
- Expected jobs: `240`.

## Paths

- Large output root:
  `/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1`
- Review root:
  `/homebase/code_sync/inrorspline/experiments/2026-06-25_oasis_seed3_first10_alpha_sweep`
- Host large output equivalent:
  `/data/hengjie/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1`

## Review-Gated Commands

Run only after approval, inside Docker container `py310sitreg_v1_test1`:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_first10_be_2500e.py --dry-run
```

After reviewing the dry-run manifest and commands:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_first10_be_2500e.py
```

If aggregation needs to be rerun after completed jobs:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/aggregate_first10_population.py
```

## Expected Review Outputs

- `summary_runs.csv/json`
- `population_summary.csv/json`
- `population_label_metrics.csv`
- `case_metrics_all.csv`
- `regularity_metrics_all.csv`
- `be_metrics_all.csv`
- `diffusion_metrics_all.csv`
- `loss_history_all.csv`
- `loss_history_population.csv`
- `population_arc_primary.pdf`
- `population_pareto_contact_sheet.pdf`
- `loss_curves_population.pdf`
- `review_report.md`
