---
doc_type: workflow
status: active
---

# Server Sync Workflow

Only `rsync` requires a mandatory impact explanation. Git/GitHub remains the source of truth for code; rsync is for one-way mirrors or controlled data transfer.

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
