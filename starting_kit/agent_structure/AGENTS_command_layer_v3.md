---
doc_type: agents_command_layer
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
  - AGENTS.md
  - docs/00_index.md
  - docs/workflows/server_sync_workflow.md
  - docs/workflows/docker_gpu_workflow.md
---

# AGENTS.md Command Layer: New Multi-Server Research Project

Paste this near the top of the root `AGENTS.md`, after `Start Here`. It gives Codex a concise command interface while the repo, servers, datasets, Docker setup, and anonymous-release path are still being adapted.

## Project Operating Object

```text
PROJECT_OBJECT = {
  phase: "new_project_adaptation",
  repo_source_of_truth: "git/GitHub for code, docs, configs, tests, public-review artifacts",
  code_write_model: "single-writer primary development repo",
  worker_model: "GPU workers are execution mirrors unless explicitly promoted",
  datasets: ["brain", "mountain"],
  data_source_of_truth: "server-local DATA_ROOT folders, never Git",
  docker_model: "container gets separate repo/data/scratch mounts",
  repo_mount_in_container: "/workspace/repo",
  data_mount_in_container: "/workspace/data",
  scratch_mount_in_container: "/workspace/scratch",
  raw_data_mode: "read-only by default",
  rsync_policy: "explain impact first, dry-run first, real sync only after approval",
  docker_policy: "normal requested operation; no special impact explanation required",
  public_review_policy: "anonymous-safe code and docs from day one"
}
```

## Default Path Model

Do not hard-code real private paths in public docs. Put server-specific values in `docs/00_index.md` if review-safe, or in ignored files such as `private/server_map.md`.

```bash
export PROJECT_SLUG=TODO_PROJECT_SLUG
export REPO_ROOT="$HOME/$PROJECT_SLUG"              # Git checkout on each server
export DATA_ROOT="/data/$USER/$PROJECT_SLUG"        # dedicated data root on each server
export SCRATCH_ROOT="/scratch/$USER/$PROJECT_SLUG"  # large outputs/checkpoints/logs
export BRAIN_DATA="$DATA_ROOT/brain"
export MOUNTAIN_DATA="$DATA_ROOT/mountain"
```

Expected host layout:

```text
$REPO_ROOT/                 # Git-managed code/docs/configs/tests
$DATA_ROOT/brain/           # brain dataset, outside Git
$DATA_ROOT/mountain/        # mountain dataset, outside Git
$DATA_ROOT/manifests/       # manifests/checksums for data integrity
$SCRATCH_ROOT/runs/         # large experiment outputs, outside Git
$SCRATCH_ROOT/tmp/          # temporary files, outside Git
```

Expected container layout:

```text
/workspace/repo             # mounted from $REPO_ROOT
/workspace/data             # mounted from $DATA_ROOT; raw data read-only by default
/workspace/scratch          # mounted from $SCRATCH_ROOT; outputs/checkpoints/logs
```

Docker two-mount-plus-scratch template:

```bash
docker run --rm -it --gpus all \
  --ipc=host \
  -v "$REPO_ROOT":/workspace/repo:rw \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$SCRATCH_ROOT":/workspace/scratch:rw \
  -w /workspace/repo \
  "$IMAGE" \
  python scripts/experiments/TODO_RUN_SCRIPT.py --help
```

Use `:ro` for raw data unless a derived-data write workflow is explicitly approved.

## Command Interface

### `[adapt project]`

Use when project assumptions change.

Codex must:

- Read `AGENTS.md`, `docs/00_index.md`, docs/workflows, `.gitignore`, `.rsync-exclude`, Docker files, scripts, configs, and tests if present.
- Treat the repo as new and evolving; use TODO placeholders rather than inventing server names, dataset schemas, host paths, or paper claims.
- Preserve reusable old-project patterns: plan mode, commit gating, experiment registry, technical logs, documentation updates, smoke tests, and exclusions for raw data, outputs, checkpoints, credentials, Docker artifacts, and large files.
- Classify old project-specific content as `KEEP`, `ADAPT`, `REMOVE`, or `LEGACY-NOT-YET-REMOVED` before deleting it.
- Return a plan first. Do not edit until approved.

Prompt:

```text
[adapt project]
Adapt AGENTS.md and the docs scaffold to these new constraints:
- <paste changes>
Return a plan only. For each proposed change, list the exact file, whether it is KEEP/ADAPT/REMOVE/CREATE, and why. Do not run code, rsync, Docker, or data commands.
```

### `[adapt AGENTS.md]`

Use for directly revising the root guide.

Codex must produce this adaptation report before editing:

```text
New rules:
Old rules to keep:
Old rules to adapt:
Old rules to remove:
Legacy rules not yet removed:
Unknown TODOs:
Files affected:
Verification after edit:
```

Prompt:

```text
[adapt AGENTS.md]
Incorporate the following changes into AGENTS.md. Keep the project new/adaptable. Do not silently delete legacy content; first classify it as KEEP, ADAPT, REMOVE, or LEGACY-NOT-YET-REMOVED. Return the plan before editing.

Changes:
- <paste changes>
```

### `[sync-code plan]`

Use for one-way code mirroring from the canonical primary repo to a worker.

Codex must:

- Confirm this is code sync, not dataset sync.
- Use Git/GitHub as the source of truth for code.
- Use one-way `rsync` only to mirror a clean repo to a disposable worker.
- Never use bidirectional rsync.
- Explain rsync impact first using the rsync briefing format below.
- Provide a dry-run command first.
- Wait for approval before real rsync.

Dry-run template:

```bash
SOURCE_REPO="$REPO_ROOT"
TARGET_HOST=TODO_GPU_WORKER
TARGET_REPO=/home/TODO_USER/TODO_PROJECT_SLUG

rsync -azP --itemize-changes --delete \
  --exclude-from=.rsync-exclude \
  "$SOURCE_REPO"/ "$TARGET_HOST":"$TARGET_REPO"/ \
  --dry-run
```

Real sync only after the dry-run has been inspected:

```bash
rsync -azP --itemize-changes --delete \
  --exclude-from=.rsync-exclude \
  "$SOURCE_REPO"/ "$TARGET_HOST":"$TARGET_REPO"/
```

### `[sync-data plan]`

Use for one-way transfer of `brain` or `mountain` between dedicated data roots.

Codex must:

- Confirm dataset name: `brain`, `mountain`, or both.
- Confirm source host/path and target host/path.
- Treat raw data as precious. Default to copy/update mode without `--delete`.
- Explain rsync impact first using the rsync briefing format below.
- Use `--dry-run` first.
- Generate or compare manifests before claiming transfer success.
- Never write raw data into the Git repo.
- Never use destructive raw-data mirroring unless explicitly approved after a manifest/checksum plan.

Dry-run template:

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

Real copy, after approval:

```bash
rsync -aP --info=progress2 --partial --itemize-changes \
  "$SRC_HOST":"$SRC_DATA"/ \
  "$DST_HOST":"$DST_DATA"/
```

No `--delete` for raw datasets by default.

### `[rsync explain]`

Use before any rsync command, whether for code, data, Docker image archives, logs, results, or patches.

Rsync briefing format:

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

### `[docker run]`

Use for Docker build/run/test/training commands. Docker does **not** require the rsync-style impact briefing.

Codex should still keep commands explicit and reproducible:

```bash
docker run --rm -it --gpus all \
  --ipc=host \
  -v "$REPO_ROOT":/workspace/repo:rw \
  -v "$DATA_ROOT":/workspace/data:ro \
  -v "$SCRATCH_ROOT":/workspace/scratch:rw \
  -w /workspace/repo \
  "$IMAGE" \
  <command-inside-container>
```

For long training jobs, prefer repo read-only after code is finalized:

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

### `[docker image transfer]`

Use for Docker image movement. Docker registry push/pull does not require rsync briefing. If image movement uses `rsync` for an image archive, the `rsync explain` rule applies to that rsync step only.

Registry path:

```bash
docker build -f docker/dev-gpu/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
ssh TODO_GPU_WORKER "docker pull '$IMAGE'"
```

No-registry path:

```bash
docker save "$IMAGE" | zstd -T0 -19 -o /tmp/${PROJECT_SLUG}_${GIT_SHA}.tar.zst
rsync -azP /tmp/${PROJECT_SLUG}_${GIT_SHA}.tar.zst TODO_GPU_WORKER:/tmp/
ssh TODO_GPU_WORKER "zstd -dc /tmp/${PROJECT_SLUG}_${GIT_SHA}.tar.zst | docker load"
```

Do not commit image archives.

### `[experiment launch plan]`

Use before real runs.

Codex must confirm:

- Experiment registry entry.
- Git commit and branch.
- Docker image tag or digest.
- Dataset roots and manifests.
- Output root under scratch or ignored outputs.
- Smoke test command.
- Exact launch command.

Do not launch if the user requested plan mode. Outside plan mode, Docker itself does not require an additional impact explanation.

### `[anonymous release audit]`

Use continuously, not only at the end.

Codex must inspect review-facing files for private names, emails, usernames, hostnames, absolute private paths, W&B entities, API keys, tokens, grant acknowledgments, institution identifiers, and hidden metadata. Produce or update `public/ANONYMITY_AUDIT.md` and `public/release_manifest.md`.

### `[prepare commit]` and `[summarize and commit]`

Keep the inherited behavior: inspect status/diffs/untracked files, identify excluded artifacts, propose pathspec-limited staging commands, and wait for approval before staging or committing.

## Rsync-Only Safety Gate

Only rsync requires the mandatory impact briefing. This applies to rsync for code, data, outputs, image archives, patches, logs, or any remote path.

Rules:

- Always state source and target.
- Always state whether `--delete` is present.
- Always dry-run first when `--delete` is used or when syncing datasets.
- Never use bidirectional rsync for source code.
- Never use `--delete` on raw datasets by default.
- Never sync raw data into the Git repo.
- Log meaningful real syncs in `logs/YYYY-MM.md` or `outputs/inspection/`.

High-risk rsync cases:

```text
rsync with --delete
rsync involving DATA_ROOT
rsync involving SCRATCH_ROOT
rsync from worker back to primary repo
rsync into a path that already contains data
rsync of Docker image archives, checkpoints, or outputs
```

Approval examples:

```text
Approved: run the rsync dry-run only.
Approved: run the real code rsync after the dry-run shown above.
Approved: run the real data rsync without --delete.
```

## Data Integrity Checklist

Fast manifest:

```bash
mkdir -p "$DATA_ROOT/manifests"
find "$BRAIN_DATA" -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort > "$DATA_ROOT/manifests/brain_manifest_$(hostname)_$(date +%F).tsv"
find "$MOUNTAIN_DATA" -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort > "$DATA_ROOT/manifests/mountain_manifest_$(hostname)_$(date +%F).tsv"
```

Checksum manifest, slower:

```bash
find "$BRAIN_DATA" -type f -print0 | sort -z | xargs -0 sha256sum > "$DATA_ROOT/manifests/brain_sha256_$(hostname)_$(date +%F).txt"
```

## Minimal Prompt to Initialize the Repo

```text
[adapt project]
This is a new multi-server GPU research repo. Keep Git/GitHub as the source of truth for code/docs/configs/tests. Keep raw brain and mountain datasets outside Git under dedicated per-server DATA_ROOT folders. Docker must mount the repo and data as separate volumes: /workspace/repo and /workspace/data, with raw data read-only by default, plus /workspace/scratch for outputs. Add a safety gate only for rsync: every rsync must explain source, target, direction, delete behavior, exclusions, dry-run, real command, and log/manifest plan. Docker operations do not require the rsync-style impact explanation. No bidirectional source-code sync; no raw data in the repo. Build anonymous-review-safe code continuously. Return a plan before editing.
```
