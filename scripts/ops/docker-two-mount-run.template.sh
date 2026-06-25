#!/usr/bin/env bash
# Template only. Review values before running. Do not commit server-private values.
set -euo pipefail

: "${PROJECT_SLUG:=inrorspline}"
: "${REPO_ROOT:=$HOME/code_sync/$PROJECT_SLUG}"
: "${DATA_ROOT:=/data/$USER/$PROJECT_SLUG}"
: "${SCRATCH_ROOT:=/scratch/$USER/$PROJECT_SLUG}"
: "${IMAGE:=TODO_REGISTRY/inrorspline:dev-gpu-TODO}"
: "${COMMAND:=bash}"

mkdir -p "$SCRATCH_ROOT"

cat <<EOF
Docker launch
  image:         $IMAGE
  repo mount:    $REPO_ROOT -> /workspace/repo:rw
  data mount:    $DATA_ROOT -> /workspace/data:ro
  scratch mount: $SCRATCH_ROOT -> /workspace/scratch:rw
  workdir:       /workspace/repo
  command:       $COMMAND
EOF

docker run --rm -it --gpus all \
  --ipc=host \
  -v "$REPO_ROOT":/workspace/repo:rw \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$SCRATCH_ROOT":/workspace/scratch:rw \
  -w /workspace/repo \
  "$IMAGE" \
  $COMMAND
