---
doc_type: codex_workflow
status: active
---

# Codex Workflow

## Default Behavior

- Read `AGENTS.md` and relevant docs before editing.
- Keep changes scoped.
- Do not run rsync without the required briefing and approval.
- Do not launch Docker or long experiments in plan mode.
- Do not commit unless explicitly asked.
- Treat `starting_kit/` as source material, not final project structure.

## Planning Commands

Use bracket commands from `AGENTS.md` for repeatable workflows, especially:

- `[adapt project]`
- `[experiment launch plan]`
- `[sync-code plan]`
- `[sync-data plan]`
- `[anonymous release audit]`
- `[prepare commit]`

## File Safety

Never add raw data, checkpoints, generated outputs, Docker image archives, credentials, private hostnames, private usernames, W&B entities, or identity-breaking metadata.
