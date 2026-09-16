---
doc_type: project_index
status: active
---

# Project Index

## Current Phase

`inrorspline` is the real repo root and now uses a source-only Git worktree.
The external artifact-store migration completed on 2026-09-03 after its
metadata, checksum, inventory, and writer-free final-delta checks passed. The
former source data remain rollback-only through 2026-10-03; no deletion is
authorized during that retention period.

## Key Docs

- `README.md`: top-level orientation.
- `AGENTS.md`: repo operating rules for Codex and other agents.
- `docs/01_project_brief.md`: scientific framing.
- `docs/02_literature_review.md`: literature synthesis map.
- `docs/03_methods.md`: method and protocol plan.
- `docs/implementation/`: code-derived implementation notes for method architecture, training losses, deformation math, and paper claims.
- `docs/04_experiment_registry.md`: experiment ledger.
- `docs/migrations/artifact_store_2026-09-02.md`: completed external
  artifact-store migration, verification record, and retention policy.
- `docs/05_results_summary.md`: current evidence map.
- `docs/literature/`: paper reading notes and external-code links.
- `docs/standards/`: experiment preflight standards for fair comparisons.
- `docs/todos/camera_ready_checklist.md`: blocking experiment, figure, and
  manuscript checks for the camera-ready version.
- `docs/reports/current_results_audit_and_plan.md`: detailed carried-forward audit.
- `docs/reports/reorganized_paper_points.md`: paper narrative plan.
- `docs/reports/protocol_warnings/`: warnings that must be read before LungCT comparison.

## Path Model

```bash
export PROJECT_SLUG=inrorspline
export REPO_ROOT="$HOME/code_sync/$PROJECT_SLUG"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:?set ARTIFACT_ROOT to the external artifact store}"
export ARTIFACT_RUN_ROOT="$ARTIFACT_ROOT/runs"
export ARTIFACT_REVIEW_ROOT="$ARTIFACT_ROOT/repo_experiments"
export ARTIFACT_MANIFEST_ROOT="$ARTIFACT_ROOT/migration_manifests"
```

Do not place raw datasets or generated artifacts under this repo. Use
`ARTIFACT_ROOT` for all project run/review outputs and migrated project data;
keep the host-specific value in ignored local configuration. The repo retains
source code, documentation, specifications, and output-free notebook sources.

## Immediate Next Decisions

- Maintain the external artifact-store retention window through 2026-10-03;
  any later source-data deletion needs a fresh checksum comparison and separate
  approval for the named subtree.
- Keep actual server and storage mappings only in ignored `private/` files.
- Keep DIR-Lab raw data and named preprocessing protocols under `LUNGCT_DATA`.
- Decide when to add a Python package/test scaffold.
- Convert the carried-forward absolute source paths in `docs/reports/` into repo-relative or private manifests where appropriate.
