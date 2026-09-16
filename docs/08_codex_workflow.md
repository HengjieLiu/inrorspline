---
doc_type: codex_workflow
status: active
---

# Codex Workflow

## Default Behavior

- Read `AGENTS.md` and relevant docs before editing.
- Keep changes scoped.
- Update `docs/implementation/` when code review, paper writing, or experiment
  work reveals durable implementation details about method architecture,
  training losses, deformation math, or paper-facing claims.
- Do not run rsync without the required briefing and approval.
- Do not launch Docker or long experiments in plan mode.
- Do not commit unless explicitly asked.
- Treat `starting_kit/` as source material, not final project structure.
- Keep the Git worktree source-only. Configure `ARTIFACT_ROOT` outside the repo
  for all generated run and review material; do not add broad `experiments/`
  ignores that would conceal accidental local outputs.
- Keep source notebooks output-free in Git. Execute review copies and write
  figures, tables, reports, monitors, and embedded notebook output beneath
  `${ARTIFACT_ROOT}/repo_experiments/<experiment-key>/`.
- Before a source-only audit or commit, run
  `python -B scripts/experiments/check_tracked_notebook_outputs.py`; it must
  report no Git-tracked notebook cell outputs.

## Planning Commands

Use bracket commands from `AGENTS.md` for repeatable workflows, especially:

- `[adapt project]`
- `[experiment launch plan]`
- `[sync-code plan]`
- `[sync-data plan]`
- `[artifact-store migration plan]`
- `[anonymous release audit]`
- `[prepare commit]`

## File Safety

Never add raw data, checkpoints, generated outputs (including figures, result
tables, reports, monitor files, or executed notebooks), Docker image archives,
credentials, private hostnames, private usernames, W&B entities, or
identity-breaking metadata. For an artifact migration, copy first, validate
with a final writer-free checksum pass, retain the source for 30 days, and
obtain separate approval before deletion.
