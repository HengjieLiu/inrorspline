# inrorspline

Research scaffold for a reproducible study of implicit neural representation (INR), spline, and direct control-point deformation priors in medical image registration.

The working thesis is that continuous/off-grid deformation evaluation is not enough to explain registration behavior. The practical prior induced by parameterization, optimization, regularization, and motion regime matters. The current project focuses on OASIS brain registration and LungCT/4DCT registration, with alignment-regularity Pareto evaluation as the main comparison style.

## Start Here

- [docs/00_index.md](docs/00_index.md): project map and current status.
- [docs/01_project_brief.md](docs/01_project_brief.md): scientific goal and claims.
- [docs/03_methods.md](docs/03_methods.md): method/protocol definitions.
- [docs/implementation/](docs/implementation/): code-derived implementation notes for methods, training losses, deformation math, and paper claims.
- [docs/04_experiment_registry.md](docs/04_experiment_registry.md): planned and real experiment ledger.
- [docs/reports/current_results_audit_and_plan.md](docs/reports/current_results_audit_and_plan.md): carried-forward audit and priority plan.
- [docs/reports/protocol_warnings/lungct_idir_ptvreg_protocol_bias.md](docs/reports/protocol_warnings/lungct_idir_ptvreg_protocol_bias.md): required warning before interpreting LungCT comparisons.

## Repo Model

Code, docs, configs, tests, and public-review artifacts live in Git. Large datasets, raw outputs, checkpoints, and server-private paths do not.

Default path model:

```bash
export PROJECT_SLUG=inrorspline
export REPO_ROOT="$HOME/code_sync/$PROJECT_SLUG"
export DATA_ROOT="/data/$USER/$PROJECT_SLUG"
export SCRATCH_ROOT="/scratch/$USER/$PROJECT_SLUG"
```

Docker should mount these as separate volumes:

```text
/workspace/repo      # code repo
/workspace/data      # dataset root, read-only by default
/workspace/scratch   # runs, checkpoints, logs
```

## Current Status

This repository has just been scaffolded from `starting_kit/`. It intentionally does not yet include a Python package scaffold, real experiment configs, server-specific launch scripts, raw data, checkpoints, or generated result artifacts.
