# Codex Execution Spec

## Goal

Implement and execute the v003 OASIS fixed-mask Panel A diagnostic refresh,
then regenerate the complete population diagnostic figure family without
overwriting v001 or v002 artifacts.

## Commands

Syntax checks and runs should execute inside the default Docker container:

```bash
docker exec py310sitreg_v1_test1 python -m py_compile \
  ${REPO_ROOT_IN_CONTAINER}/scripts/experiments/oasis_alpha_sweep/run_one.py \
  ${REPO_ROOT_IN_CONTAINER}/scripts/experiments/fit_modes_population/launch_oasis_fixedmask_panelA_curves.py \
  ${REPO_ROOT_IN_CONTAINER}/scripts/experiments/fit_modes_population/render_masked_panelA_both_datasets_diagnostics.py
```

Dry-run:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT_IN_CONTAINER}/scripts/experiments/fit_modes_population/launch_oasis_fixedmask_panelA_curves.py \
  --dry-run --gpus 0,1,2,3
```

Smoke:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT_IN_CONTAINER}/scripts/experiments/fit_modes_population/launch_oasis_fixedmask_panelA_curves.py \
  --smoke --gpus 0,1,2,3
```

Full run:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT_IN_CONTAINER}/scripts/experiments/fit_modes_population/launch_oasis_fixedmask_panelA_curves.py \
  --resume --gpus 0,1,2,3
```

Render v003:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT_IN_CONTAINER}/scripts/experiments/fit_modes_population/render_masked_panelA_both_datasets_diagnostics.py
```

## Non-overwrite policy

The launcher reuses exact-compatible outputs and otherwise chooses versioned
run names for incompatible existing directories.  v001 and v002 paper artifacts
are read-only inputs for this workflow.
