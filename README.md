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

Code, docs, configs, tests, hand-authored experiment specifications, and
output-free notebook sources live in Git. Generated experiment and review
artifacts do not: figures, PDFs, result tables, reports, monitors, checkpoints,
dense fields, and executed notebooks belong in the external artifact store.

Default path model:

```bash
export PROJECT_SLUG=inrorspline
export REPO_ROOT="$HOME/code_sync/$PROJECT_SLUG"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:?set ARTIFACT_ROOT to the external artifact store}"
export ARTIFACT_RUN_ROOT="$ARTIFACT_ROOT/runs"
export ARTIFACT_REVIEW_ROOT="$ARTIFACT_ROOT/repo_experiments"
```

The actual host location of `ARTIFACT_ROOT` is private operational context and
is recorded only in ignored local configuration. Its expected layout is:

```text
${ARTIFACT_ROOT}/runs/                 # run outputs, checkpoints, fields, logs
${ARTIFACT_ROOT}/ptvreg/               # migrated pTVReg/project artifacts
${ARTIFACT_ROOT}/prepared/             # prepared project data
${ARTIFACT_ROOT}/dual_author_data/     # migrated project data
${ARTIFACT_ROOT}/repo_experiments/     # review artifacts and executed notebooks
${ARTIFACT_ROOT}/migration_manifests/  # migration inventories, checksums, logs
```

The repository deliberately keeps a narrow `.gitignore`; accidental generated
files under `experiments/` should be visible and moved out, not broadly ignored.

Docker should mount these as separate volumes:

```text
/workspace/repo      # code repo
/workspace/data      # optional raw-data mounts, read-only by default
/database/inrorspline # compatibility mount for legacy run paths
```

## Current Status

The external-artifact-store migration completed on 2026-09-03 after the
metadata canary, writer-free final delta, checksum comparison, and inventories
passed. The former source data remain rollback-only through 2026-10-03; no
source-data deletion is authorized before then. See
[`docs/migrations/artifact_store_2026-09-02.md`](docs/migrations/artifact_store_2026-09-02.md).
