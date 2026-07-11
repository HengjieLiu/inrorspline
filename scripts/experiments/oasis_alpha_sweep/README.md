# OASIS Alpha Sweep Scripts

Repo-local OASIS alpha-sweep runner for the seed-3 shuffled BrainMR protocol.

These scripts intentionally do not import or execute legacy task9 experiment
scripts. They load OASIS NIfTI files directly, apply the same OASIS orientation
conversion used by prior runs, and write large artifacts outside Git.

Default paths inside the Docker container:

```text
OASIS root: /database/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1
run root:   /database/inrorspline/runs/oasis_seed3_case0_prerun_be_1000e_v1
review:     experiments/2026-06-25_oasis_seed3_case0_prerun
```

Case-0 pre-run launch command from inside the container:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_prerun.py --overwrite
```

First10 population BE sweep, review-gated:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_first10_be_2500e.py --dry-run
python scripts/experiments/oasis_alpha_sweep/launch_first10_be_2500e.py
```

The first10 aggregator uses population means by method/alpha for ARC-style
curves:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/aggregate_first10_population.py
```

Saved DL/Greedy baseline evaluation for the same seed-3 first10 protocol:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/evaluate_saved_baselines_first10.py --dry-run
python scripts/experiments/oasis_alpha_sweep/evaluate_saved_baselines_first10.py
```

This writes large saved-field evaluation artifacts to
`/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1` and review
overlays to `experiments/2026-06-25_oasis_seed3_first10_saved_baselines/`.
Greedy displacement NIfTI files use a 5D channel-last storage convention and
are reoriented inside the evaluator before metric computation.

Rank-9 image-registration vs Greedy-field fitting LR sweep:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_rank009_fit_modes_lr_sweep.py --dry-run --gpus 0,1,2
python scripts/experiments/oasis_alpha_sweep/launch_rank009_fit_modes_lr_sweep.py --gpus 0,1,2 --max-concurrent 3
```

This writes large artifacts to
`/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1` and review
figures/tables to
`experiments/2026-06-26_oasis_rank009_fit_modes_lr_sweep/`.
Greedy-field fitting uses fixed-image-positive brain-mask MSE: IDIR samples
10,000 masked coordinates per epoch, while SINR/DirectCP evaluate MSE over all
masked voxels after cps4 dense-field expansion.

Rank-9 reused image-registration vs SITReg-field fitting LR sweep:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_rank009_sitreg_fit_modes_lr_sweep.py --dry-run --gpus 2,3 --max-concurrent 2
python scripts/experiments/oasis_alpha_sweep/launch_rank009_sitreg_fit_modes_lr_sweep.py --gpus 2,3 --max-concurrent 2
```

This writes the 9 new SITReg field-fit artifacts to
`/database/inrorspline/runs/oasis_rank009_sitreg_fit_modes_lr_sweep_v1`, reuses
the completed 9 image-registration runs from
`/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`, and writes
combined review figures/tables to
`experiments/2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep/`. The SITReg
target is the saved `bs4_sitreg_v0` best-Dice dense field in channel-first
`zyx` repo/RAS layout with no Greedy-style reorientation or sign flip.
