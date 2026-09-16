---
doc_type: workflow
status: active
---

# Server Sync Workflow

Only `rsync` requires a mandatory impact explanation. Git/GitHub remains the
source of truth for source code and docs; rsync is for one-way mirrors or
controlled data transfer. Generated project artifacts belong in
`${ARTIFACT_ROOT}`, never in the Git worktree.

## Rsync Briefing

```text
Rsync purpose:
Source host/path:
Target host/path:
Direction: one-way source -> target
Scope: code | brain data | lungct data | outputs | image archive | patch | other
Delete enabled: no/yes
Will create:
Will overwrite/update:
May delete:
Excluded paths/patterns:
Dry-run command:
Real command after approval:
Manifest/checksum/log plan:
Approval needed:
```

## Code Mirror Template

```bash
SOURCE_REPO="$REPO_ROOT"
TARGET_HOST=TODO_GPU_WORKER
TARGET_REPO=/home/TODO_USER/inrorspline

rsync -azP --itemize-changes --delete \
  --exclude-from=.rsync-exclude \
  "$SOURCE_REPO"/ "$TARGET_HOST":"$TARGET_REPO"/ \
  --dry-run
```

Run the real command only after dry-run review and approval.

## Data Sync Template

```bash
DATASET=brain
SRC_HOST=TODO_SOURCE_HOST
DST_HOST=TODO_TARGET_HOST
SRC_DATA=/data/TODO_USER/inrorspline/$DATASET
DST_DATA=/data/TODO_USER/inrorspline/$DATASET

ssh "$DST_HOST" "mkdir -p '$DST_DATA'"
rsync -aP --info=progress2 --partial --itemize-changes \
  "$SRC_HOST":"$SRC_DATA"/ \
  "$DST_HOST":"$DST_DATA"/ \
  --dry-run
```

Use `DATASET=lungct` for LungCT. Do not use `--delete` for raw datasets by default.

## External Artifact Store Migration

Use this workflow for a copy/check/retain migration of project artifacts to
`ARTIFACT_ROOT`. The actual host path is private; do not put it in committed
commands or review-facing docs.

Before the first real copy, confirm all of the following:

- the destination is writable, intentionally empty or inventoried, and has at
  least 1 TiB of available capacity;
- no destination or source path ambiguity remains;
- a small canary can preserve paths, bytes, file types, modes, and modification
  times at the destination filesystem's supported precision; and
- a writer-free window is scheduled for finalization.

For every top-level subtree, use archive mode with hard links, ACLs, xattrs,
and sparse files enabled, while intentionally not preserving owner or group.
Do not substitute a weaker metadata mode if the destination cannot support the
required metadata: stop and resolve that failure. `ctime`, `atime`, and birth
time are not transferable guarantees; ownership/group become the selected
transfer user.

```bash
SOURCE_SUBTREE=TODO_SOURCE_SUBTREE
DEST_SUBTREE="${ARTIFACT_ROOT}/TODO_SUBTREE"
MANIFEST_ROOT="${ARTIFACT_ROOT}/migration_manifests/TODO_MIGRATION_ID"

rsync -aHAXSx --no-o --no-g --partial --partial-dir="$MANIFEST_ROOT/partials" \
  --itemize-changes \
  "$SOURCE_SUBTREE"/ "$DEST_SUBTREE"/ \
  --dry-run
```

After dry-run review and approval, repeat the same command without `--dry-run`
and record its itemized log, source/destination inventories, and selected
metadata-canary result under `MANIFEST_ROOT`. Do not use `--delete`,
`--remove-source-files`, or `--inplace` in either the initial copy or final
delta pass.

Freeze writers before finalization, rerun the non-destructive delta copy, then
require a checksum comparison before validation:

```bash
rsync -aHAXSx --no-o --no-g --checksum --dry-run --itemize-changes \
  "$SOURCE_SUBTREE"/ "$DEST_SUBTREE"/
```

The comparison must be empty and inventories must agree on file count, total
bytes, modification times, file types, and modes. Retain each verified source
subtree unused for at least 30 days. A later deletion requires another checksum
comparison and explicit, per-subtree approval; it is not part of this workflow.
