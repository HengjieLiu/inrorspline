# Codex Execution Spec

## Objective

Extend the completed OASIS seed-3 first20 publication-figure experiment to a
first100 cohort by running seed-3 shuffled ranks `20-99` and reusing ranks
`0-19`.

## Environment

- Docker container: `py310sitreg_v1_test1`
- Repo path in container: `/homebase/code_sync/inrorspline`
- Large output root: `/database/inrorspline/runs/`
- GPU policy: use all visible GPUs, one job per GPU, unless `--max-concurrent`
  is set.

## New Output Roots

- Saved baselines:
  `/database/inrorspline/runs/oasis_seed3_ranks020_099_saved_baselines_v1`
- Single-stage sweeps:
  `/database/inrorspline/runs/oasis_seed3_ranks020_099_alpha_sweep_be_2500e_5models_v1`
- MR-D-BSCP 8-4-2:
  `/database/inrorspline/runs/oasis_seed3_ranks020_099_directcp842_lr1e3_alpha_sweep_v1`
- Review root:
  `experiments/2026-07-04_oasis_seed3_first100_extension/`

## Run Order

1. Saved baseline/reference evaluation:
   `bs1_vxm_v0_2`, `bs2_tm_v0`, `bs3_vfa_v0`, `bs4_sitreg_v0`, `greedy`.
2. MR-D-BSCP 8-4-2:
   `direct_cp_8_4_2_be_bspline_analytic_multistage`, alpha
   `0,10,100,1000,10000,100000`, lr `1e-3`, cps `8,4,2`,
   downsample `4,2,1`, stage epochs `400,800,1300`.
3. D-BSCP cps2:
   `direct_cp_cps2_be_bspline_analytic`, alpha
   `0,10,100,1000,10000,100000`, lr `1e-3`.
4. INR-BSCP cps2:
   `sinr_cps2_be_bspline_analytic`, alpha
   `0,10,100,1000,10000,100000`, lr `1e-4`.
5. INR-Dense:
   `idir_original_dense_siren_be_autograd`, alpha
   `0,0.01,0.1,1,10,100`, lr `1e-4`, hidden dim `256`,
   hidden layers `3`, omega `32`, coord batch `10000`.
6. D-BSCP cps4:
   `direct_cp_cps4_be_bspline_analytic`, alpha
   `0,10,100,1000,10000,100000`, lr `1e-3`.
7. INR-BSCP cps4:
   `sinr_cps4_be_bspline_analytic`, alpha
   `0,10,100,1000,10000,100000`, lr `1e-4`.

Common sweep settings: ranks `20-99`, epochs `2500`, eval interval `250`,
seed `1`, foreground regularity mask `fixed_image > 0`, segmentation only for
Dice/HD95/ASD.

## Commands

Dry-run the full launch matrix:

```bash
python scripts/experiments/oasis_alpha_sweep/launch_oasis_first100_extension.py --stage all --dry-run --gpus auto --max-concurrent 0
```

Launch the full matrix:

```bash
python scripts/experiments/oasis_alpha_sweep/launch_oasis_first100_extension.py --stage all --gpus auto --max-concurrent 0
```

Run only the tracker:

```bash
python scripts/experiments/oasis_alpha_sweep/monitor_oasis_first100_extension.py --watch --interval-sec 300
```

Aggregate strict first100 figures after all rows complete:

```bash
python scripts/experiments/oasis_alpha_sweep/aggregate_oasis_first100_extension.py
```
