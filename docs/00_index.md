---
doc_type: project_index
status: active
---

# Project Index

## Current Phase

`inrorspline` has been initialized as the real repo root. The repo currently contains scaffold docs and carried-forward planning reports, but no Python package scaffold, real launch configs, raw data, checkpoints, or generated result artifacts.

## Key Docs

- `README.md`: top-level orientation.
- `AGENTS.md`: repo operating rules for Codex and other agents.
- `docs/01_project_brief.md`: scientific framing.
- `docs/02_literature_review.md`: literature synthesis map.
- `docs/03_methods.md`: method and protocol plan.
- `docs/implementation/`: code-derived implementation notes for method architecture, training losses, deformation math, and paper claims.
- `docs/04_experiment_registry.md`: experiment ledger.
- `docs/05_results_summary.md`: current evidence map.
- `docs/literature/`: paper reading notes and external-code links.
- `docs/standards/`: experiment preflight standards for fair comparisons.
- `docs/reports/current_results_audit_and_plan.md`: detailed carried-forward audit.
- `docs/reports/reorganized_paper_points.md`: paper narrative plan.
- `docs/reports/protocol_warnings/`: warnings that must be read before LungCT comparison.

## Path Model

```bash
export PROJECT_SLUG=inrorspline
export REPO_ROOT="$HOME/code_sync/$PROJECT_SLUG"
export DATA_ROOT="/data/$USER/$PROJECT_SLUG"
export SCRATCH_ROOT="/scratch/$USER/$PROJECT_SLUG"
export BRAIN_DATA="$DATA_ROOT/brain"
export LUNGCT_DATA="$DATA_ROOT/lungct"
```

Do not place raw datasets under this repo. Use `DATA_ROOT` for data and `SCRATCH_ROOT` for large outputs.

## Immediate Next Decisions

- Confirm actual server map and whether it can be stored in public docs or only in ignored `private/`.
- Confirm canonical dataset folder names under `DATA_ROOT`.
- Decide when to add a Python package/test scaffold.
- Convert the carried-forward absolute source paths in `docs/reports/` into repo-relative or private manifests where appropriate.
