---
doc_type: workflow
status: active
---

# Docker GPU Workflow

Docker operations do not require the rsync-style impact briefing. Keep commands explicit and reproducible.

## Environment

```bash
export PROJECT_SLUG=inrorspline
export REPO_ROOT="$HOME/code_sync/$PROJECT_SLUG"
export DATA_ROOT="${DATA_ROOT:?set the raw-data root in local configuration}"
export SHARED_STORAGE_ROOT="${SHARED_STORAGE_ROOT:?set the shared-storage root in local configuration}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:?set the external artifact store in local configuration}"
export ARTIFACT_RUN_ROOT="$ARTIFACT_ROOT/runs"
export ARTIFACT_REVIEW_ROOT="$ARTIFACT_ROOT/repo_experiments"
export IMAGE="TODO_REGISTRY/inrorspline:dev-gpu-TODO"
```

## Smoke Template

```bash
docker run --rm -it --gpus all \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e ARTIFACT_ROOT=/database/inrorspline \
  -v "$REPO_ROOT":/workspace/repo:rw \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$SHARED_STORAGE_ROOT":/mnt/shengdata1/hengjie:ro \
  -v "$ARTIFACT_ROOT":/database/inrorspline:rw \
  -w /workspace/repo \
  "$IMAGE" \
  bash
```

## Training Template

```bash
docker run --rm --gpus all \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e ARTIFACT_ROOT=/database/inrorspline \
  -v "$REPO_ROOT":/workspace/repo:ro \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$SHARED_STORAGE_ROOT":/mnt/shengdata1/hengjie:ro \
  -v "$ARTIFACT_ROOT":/database/inrorspline:rw \
  -w /workspace/repo \
  "$IMAGE" \
  python scripts/experiments/train.py \
    --config configs/experiments/TODO.yaml \
    --data-root /workspace/data \
    --output-root /database/inrorspline/runs/TODO_RUN_ID
```

Use `--gpus all` when a runner selects a physical GPU by ID and masks it inside
the process. For OASIS on the current host, set the container-side root to
`/mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1`.

For legacy containers that already mount raw data at `/database`, retain that
mount and add the more specific `${ARTIFACT_ROOT}:/database/inrorspline:rw`
bind. Add `${SHARED_STORAGE_ROOT}:/mnt/shengdata1/hengjie:ro` when datasets
live on shared storage. `docker restart` cannot add a bind mount: preserve the
old container, recreate it with the complete mount list, validate it, and only
then use the recreated container under the default name.

Review figures, result tables, reports, live-monitor files, and executed
notebooks must use
`/database/inrorspline/repo_experiments/<experiment-key>/`.
Keep only output-free sources and hand-authored run specifications in Git.
