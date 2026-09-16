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
export ARTIFACT_ROOT="${ARTIFACT_ROOT:?set the external artifact store in local configuration}"
export ARTIFACT_RUN_ROOT="$ARTIFACT_ROOT/runs"
export ARTIFACT_REVIEW_ROOT="$ARTIFACT_ROOT/repo_experiments"
export IMAGE="TODO_REGISTRY/inrorspline:dev-gpu-TODO"
```

## Smoke Template

```bash
docker run --rm -it --gpus all \
  --ipc=host \
  -e ARTIFACT_ROOT=/database/inrorspline \
  -v "$REPO_ROOT":/workspace/repo:rw \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$ARTIFACT_ROOT":/database/inrorspline:rw \
  -w /workspace/repo \
  "$IMAGE" \
  bash
```

## Training Template

```bash
docker run --rm --gpus all \
  --ipc=host \
  -e ARTIFACT_ROOT=/database/inrorspline \
  -v "$REPO_ROOT":/workspace/repo:ro \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$ARTIFACT_ROOT":/database/inrorspline:rw \
  -w /workspace/repo \
  "$IMAGE" \
  python scripts/experiments/train.py \
    --config configs/experiments/TODO.yaml \
    --data-root /workspace/data \
    --output-root /database/inrorspline/runs/TODO_RUN_ID
```

For legacy containers that already mount raw data at `/database`, retain that
mount and add the more specific `${ARTIFACT_ROOT}:/database/inrorspline:rw`
bind. This preserves existing `/database/inrorspline/runs/...` paths without
placing generated output back into the repo.

Review figures, result tables, reports, live-monitor files, and executed
notebooks must use
`/database/inrorspline/repo_experiments/<experiment-key>/`.
Keep only output-free sources and hand-authored run specifications in Git.
