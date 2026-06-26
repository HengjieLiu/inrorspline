# OASIS Seed-3 Next10 Five-Model Alpha Sweep Execution Spec

## Objective

Run the OASIS seed-3 shuffled next10 BE alpha sweep and aggregate population
ARC-style curves. Population points are method/alpha means across ranks `10-19`;
case-level and per-label metrics remain available in CSV.

## Dataset

- Dataset: OASIS BrainMR `pair_test_200`.
- Pair order: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case ranks: `10-19`.
- Original sorted indices: `130,100,118,67,79,56,142,199,36,12`.
- Fixed/moving pairs:
  - `(0275, 0402)`
  - `(0184, 0182)`
  - `(0230, 0112)`
  - `(0117, 0036)`
  - `(0141, 0004)`
  - `(0108, 0405)`
  - `(0303, 0141)`
  - `(0411, 0339)`
  - `(0072, 0037)`
  - `(0024, 0115)`

## Matrix

- IDIR: lr `1e-4`, coordinate-autograd analytic BE, alpha
  `0,0.01,0.1,1,10,100`.
- SINR cps4: lr `1e-4`, cubic B-spline analytic BE, alpha
  `0,10,100,1000,10000,100000`.
- SINR cps2: lr `1e-4`, cubic B-spline analytic BE, alpha
  `0,10,100,1000,10000,100000`.
- DirectCP cps4: lr `1e-3`, cubic B-spline analytic BE, alpha
  `0,10,100,1000,10000,100000`.
- DirectCP cps2: lr `1e-3`, cubic B-spline analytic BE, alpha
  `0,10,100,1000,10000,100000`.
- Epochs: `2500`.
- Eval interval: `250`.
- Seed: `1`.
- Expected jobs: `300`.

## Paths

- Large output root:
  `/database/inrorspline/runs/oasis_seed3_next10_alpha_sweep_be_2500e_5models_v1`
- Review root:
  `/homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_seed3_next10_alpha_sweep`
- Host large output equivalent:
  `/data/hengjie/inrorspline/runs/oasis_seed3_next10_alpha_sweep_be_2500e_5models_v1`

## Launch

The launcher runs one method phase at a time. Within each phase, the `10 x 6`
case/alpha jobs are served through a dynamic four-GPU queue.

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_next10_5models_be_2500e.py --dry-run
python scripts/experiments/oasis_alpha_sweep/launch_next10_5models_be_2500e.py
```

The real launch should run through `nohup docker exec` from the host so an SSH
drop does not interrupt the container process.

