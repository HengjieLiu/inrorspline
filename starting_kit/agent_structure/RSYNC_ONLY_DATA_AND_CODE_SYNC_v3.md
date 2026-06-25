---
doc_type: workflow
created: 2026-06-25
updated: 2026-06-25
status: active_draft
visibility: internal
public:
  initial_creation_date: 2026-06-25
  first_upload_date: null
  latest_public_update_date: null
  anonymous_release_status: not_applicable
related_code:
  - .rsync-exclude
  - docker/dev-gpu/compose.yaml
  - scripts/ops/
---

# Rsync-Only Safety + Two-Mount Docker Runbook

The repo separates three spaces:

1. **Code**: Git-managed checkout under `REPO_ROOT`.
2. **Data**: large `brain` and `mountain` datasets under `DATA_ROOT`, outside Git.
3. **Outputs**: runs, checkpoints, logs, and temporary files under `SCRATCH_ROOT` or ignored `outputs/`.

Only `rsync` requires an impact explanation. Docker build/run/push/pull commands do not need the rsync-style briefing.

## Environment Variables

```bash
export PROJECT_SLUG=TODO_PROJECT_SLUG
export REPO_ROOT="$HOME/$PROJECT_SLUG"
export DATA_ROOT="/data/$USER/$PROJECT_SLUG"
export SCRATCH_ROOT="/scratch/$USER/$PROJECT_SLUG"
export BRAIN_DATA="$DATA_ROOT/brain"
export MOUNTAIN_DATA="$DATA_ROOT/mountain"
export IMAGE="TODO_REGISTRY/TODO_PROJECT_SLUG:dev-gpu-TODO_DATE-TODO_GIT_SHA"
```

## Docker: Separate Code, Data, and Scratch Mounts

Docker operations do not require an impact briefing, but commands should still be explicit and reproducible.

Smoke/development:

```bash
docker run --rm -it --gpus all \
  --ipc=host \
  -v "$REPO_ROOT":/workspace/repo:rw \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$SCRATCH_ROOT":/workspace/scratch:rw \
  -w /workspace/repo \
  "$IMAGE" \
  python -m pytest -q
```

Training, with repo read-only after code is finalized:

```bash
docker run --rm --gpus all \
  --ipc=host \
  -v "$REPO_ROOT":/workspace/repo:ro \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$SCRATCH_ROOT":/workspace/scratch:rw \
  -w /workspace/repo \
  "$IMAGE" \
  python scripts/experiments/train.py \
    --config configs/experiments/TODO.yaml \
    --data-root /workspace/data \
    --output-root /workspace/scratch/runs/TODO_RUN_ID
```

## Rsync Briefing Required Before Any Rsync

```text
Rsync purpose:
Source host/path:
Target host/path:
Direction: one-way source -> target
Scope: code | brain data | mountain data | outputs | image archive | patch | other
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

## Code Sync: Primary to Worker

Use only from the canonical primary repo to disposable worker mirrors.

```bash
SOURCE_REPO="$REPO_ROOT"
TARGET_HOST=TODO_GPU_WORKER
TARGET_REPO=/home/TODO_USER/TODO_PROJECT_SLUG

rsync -azP --itemize-changes --delete \
  --exclude-from=.rsync-exclude \
  "$SOURCE_REPO"/ "$TARGET_HOST":"$TARGET_REPO"/ \
  --dry-run
```

After inspecting the dry-run and receiving approval:

```bash
rsync -azP --itemize-changes --delete \
  --exclude-from=.rsync-exclude \
  "$SOURCE_REPO"/ "$TARGET_HOST":"$TARGET_REPO"/
```

Do not run bidirectional code sync. If a worker has a useful fix, create a patch and review/apply it on the primary repo.

## Data Sync: Dedicated DATA_ROOT to Dedicated DATA_ROOT

Raw data transfer defaults to non-destructive copy/update. No `--delete` by default.

```bash
DATASET=brain
SRC_HOST=TODO_SOURCE_HOST
DST_HOST=TODO_TARGET_HOST
SRC_DATA=/data/TODO_USER/TODO_PROJECT_SLUG/$DATASET
DST_DATA=/data/TODO_USER/TODO_PROJECT_SLUG/$DATASET

ssh "$DST_HOST" "mkdir -p '$DST_DATA'"
rsync -aP --info=progress2 --partial --itemize-changes \
  "$SRC_HOST":"$SRC_DATA"/ \
  "$DST_HOST":"$DST_DATA"/ \
  --dry-run
```

After approval:

```bash
rsync -aP --info=progress2 --partial --itemize-changes \
  "$SRC_HOST":"$SRC_DATA"/ \
  "$DST_HOST":"$DST_DATA"/
```

For `mountain`, set `DATASET=mountain`.

## Manifests Before/After Data Sync

Fast manifest:

```bash
mkdir -p "$DATA_ROOT/manifests"
find "$DATA_ROOT/brain" -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort > "$DATA_ROOT/manifests/brain_manifest_$(hostname)_$(date +%F).tsv"
find "$DATA_ROOT/mountain" -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort > "$DATA_ROOT/manifests/mountain_manifest_$(hostname)_$(date +%F).tsv"
```

Checksum manifest, slower:

```bash
find "$DATA_ROOT/brain" -type f -print0 | sort -z | xargs -0 sha256sum > "$DATA_ROOT/manifests/brain_sha256_$(hostname)_$(date +%F).txt"
```

## Required Log Entry After Real Rsync

```text
Date:
Operator/agent:
Purpose:
Source host/path:
Target host/path:
Delete enabled: no/yes
Dry-run summary:
Real command:
Manifest before:
Manifest after:
Known issues:
```
