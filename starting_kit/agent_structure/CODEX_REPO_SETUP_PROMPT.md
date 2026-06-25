---
doc_type: codex_prompt
created: 2026-06-25
updated: 2026-06-25
status: active
visibility: internal
public:
  initial_creation_date: null
  first_upload_date: null
  latest_public_update_date: null
  anonymous_release_status: not_applicable
---

# Codex Repo Setup Prompt v3

Paste this into Codex from the repository root.

```text
[adapt project]
Read AGENTS.md first. Then inspect the repository tree, git status, docs, .gitignore, .rsync-exclude, Docker files, scripts, configs, and tests.

This is a new time-sensitive multi-server GPU research repo. The project has two large datasets named brain and mountain. Code/docs/configs/tests are managed by Git/GitHub. Datasets live outside Git under a dedicated DATA_ROOT on each server. Docker must mount the code repo and dataset root as separate volumes:

- host REPO_ROOT -> container /workspace/repo
- host DATA_ROOT -> container /workspace/data, read-only by default
- host SCRATCH_ROOT -> container /workspace/scratch, read-write for outputs/checkpoints/logs

Safety rule correction:
- Docker operations do not require a special impact explanation.
- Only rsync requires a mandatory impact explanation.
- Before any rsync, explain source, target, direction, scope, delete behavior, exclusions, dry-run command, real command, and manifest/log plan.
- Always dry-run first for dataset rsync or rsync with --delete.
- Do not use bidirectional rsync for source code.
- Do not use --delete on raw datasets unless explicitly approved after a manifest/checksum plan.

Other repo rules:
- Return a plan before editing.
- Do not run rsync during planning.
- Do not commit raw data, checkpoints, generated outputs, Docker image archives, private hostnames, credentials, or identity-breaking metadata.
- Treat old project-specific rules as KEEP, ADAPT, REMOVE, or LEGACY-NOT-YET-REMOVED before deleting them.
- Keep anonymous-review-safe code and docs ready from the beginning.

Return:
1. Current repo summary.
2. Project assumptions and unknown TODOs.
3. Files/folders to create or modify.
4. AGENTS.md adaptation plan, including command layer.
5. Docs scaffold plan.
6. Server map and path model plan.
7. Code sync workflow plan.
8. Data sync workflow plan for brain and mountain.
9. Docker two-mount workflow plan.
10. Experiment registry and technical log plan.
11. Anonymous release plan.
12. .gitignore and .rsync-exclude plan.
13. Verification/smoke-test plan.
14. Exact commands that would run only after approval.

Do not implement until I approve the plan.
```

After approving the plan, paste:

```text
Implement the approved scaffold only. Keep diffs scoped. Create or update AGENTS.md, docs, workflow templates, .gitignore, .rsync-exclude, Docker docs/templates, and minimal smoke-testable scripts. Do not add raw data, outputs, checkpoints, credentials, Docker image archives, or private host-specific files. Do not run rsync. Docker commands may be added as templates but should not be launched unless explicitly requested. After edits, show git status, changed files, excluded files, and recommended verification commands. Do not commit.
```
