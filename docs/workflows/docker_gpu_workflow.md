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
export DATA_ROOT="/data/$USER/$PROJECT_SLUG"
export SCRATCH_ROOT="/scratch/$USER/$PROJECT_SLUG"
export IMAGE="TODO_REGISTRY/inrorspline:dev-gpu-TODO"
```

## Smoke Template

```bash
docker run --rm -it --gpus all \
  --ipc=host \
  -v "$REPO_ROOT":/workspace/repo:rw \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$SCRATCH_ROOT":/workspace/scratch:rw \
  -w /workspace/repo \
  "$IMAGE" \
  bash
```

## Training Template

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

This repo does not yet include a Dockerfile or Python training entry point.
