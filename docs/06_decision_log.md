---
doc_type: decision_log
status: active
---

# Decision Log

## 2026-06-25: Promote `inrorspline` To Real Repo Root

Decision:

- `/home/hengjie/code_sync/inrorspline` is the real Git repo root.
- `starting_kit/` remains as source material for now.
- Root scaffold is created without `pyproject.toml` or a Python package/test scaffold.

Rationale:

- The immediate need is governance, documentation, protocol safety, and reproducible workflow structure before implementation or GPU runs.

## Pending Decisions

- Real server map and where to store it.
- Final public dataset handles and path conventions.
- Whether to add a Python package scaffold, and which tooling to use.
- Which OASIS alpha/LR grid to lock for the first seed-3 run.
