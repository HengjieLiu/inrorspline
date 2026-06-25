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

Pre-run launch command from inside the container:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_prerun.py --overwrite
```
