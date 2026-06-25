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
