---
doc_type: codex_operating_guide
project_slug: TODO_PROJECT_SLUG
created: 2026-06-25
updated: 2026-06-25
status: draft
visibility: internal-first_public-review-ready
public:
  initial_creation_date: 2026-06-25
  first_upload_date: null
  latest_public_update_date: null
  anonymous_release_status: not_started
related_code:
  - AGENTS.md
  - docs/00_index.md
  - docs/workflows/server_sync_workflow.md
  - docs/workflows/docker_gpu_workflow.md
  - docs/workflows/anonymous_release_workflow.md
---

# AGENTS.md - Research HPC Codebase Instructions

This file is the operating guide for Codex and human collaborators working on this repository. It is optimized for a new, fast-moving research project with multiple GPU Linux servers, two large datasets, Dockerized experiments, technical reports, and a parallel anonymous-review code release.

## Start Here

Before coding, syncing, dataset movement, Docker work, paper writing, or experiment execution, read:

- `docs/00_index.md` for the project map, server map, data roots, and current status.
- `docs/01_project_brief.md` for the scientific goal and expected claims.
- `docs/02_literature_review.md` and `docs/todos/paper_reading.md` for literature tasks.
- `docs/03_methods.md` for the current method definition.
- `docs/04_experiment_registry.md` before planning or launching real runs.
- `docs/05_results_summary.md` before writing conclusions.
- `docs/06_decision_log.md` before changing major design choices.
- `docs/07_todo.md` for lightweight global tasks.
- `docs/08_codex_workflow.md` for Codex conventions.
- `docs/workflows/server_sync_workflow.md` before using `rsync`.
- `docs/workflows/docker_gpu_workflow.md` before changing Docker definitions.
- `docs/workflows/anonymous_release_workflow.md` before touching review-facing code.

## Project Goal

This repository supports a rigorous, reproducible research project using two primary datasets:

- `brain`: TODO describe dataset.
- `mountain`: TODO describe dataset.

Reproducibility, experiment traceability, and review-safe anonymization matter as much as model performance.

## Core Operating Object

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

## Command Interface

Use these bracket commands in prompts to Codex:

```text
[adapt project]              # adapt repo instructions/docs when assumptions change; plan only
[adapt AGENTS.md]            # revise AGENTS.md after classifying legacy rules
[sync-code plan]             # one-way primary repo -> worker rsync; explain + dry-run first
[sync-data plan]             # one-way DATA_ROOT -> DATA_ROOT rsync; explain + dry-run first
[rsync explain]              # required before every rsync command
[docker run]                 # Docker build/run commands; no rsync-style impact explanation needed
[docker image transfer]      # registry push/pull, or save/load; rsync step follows rsync rule if used
[experiment launch plan]     # plan a real run with registry/log/provenance first
[anonymous release audit]    # audit review-facing code/docs for identity leaks
[prepare commit]             # summarize status and propose commit commands; do not stage/commit
[summarize and commit]       # produce commit plan first; wait for approval
```

### `[adapt project]`

Return a plan before editing. Preserve reusable old-project rules and classify old project-specific content as `KEEP`, `ADAPT`, `REMOVE`, or `LEGACY-NOT-YET-REMOVED` before deleting.

### `[adapt AGENTS.md]`

Produce this adaptation report before editing:

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

### `[prepare commit]`

Inspect status/diffs/untracked files, identify excluded artifacts, propose a commit message and pathspec-limited commands. Do not stage, commit, or push unless explicitly approved.

### `[summarize and commit]`

Return a commit plan for approval. Include topic-grouped changes, planned staged files, intentionally excluded files, submodule status, verification, one commit message, and exact commands. Stage only approved files after approval.

## Rsync-Only Safety Gate

Only `rsync` requires a mandatory impact explanation. Docker operations do not require this special briefing.

Before any rsync command, Codex must provide:

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

Rules:

- Always dry-run first for dataset rsync or any rsync with `--delete`.
- Never use bidirectional rsync for source code.
- Never use `--delete` on raw datasets by default.
- Never sync raw data into the Git repo.
- Log meaningful real syncs in `logs/YYYY-MM.md` or `outputs/inspection/`.

## Path and Mount Model

```bash
export PROJECT_SLUG=TODO_PROJECT_SLUG
export REPO_ROOT="$HOME/$PROJECT_SLUG"
export DATA_ROOT="/data/$USER/$PROJECT_SLUG"
export SCRATCH_ROOT="/scratch/$USER/$PROJECT_SLUG"
export BRAIN_DATA="$DATA_ROOT/brain"
export MOUNTAIN_DATA="$DATA_ROOT/mountain"
```

```text
$REPO_ROOT/                 # Git-managed code/docs/configs/tests
$DATA_ROOT/brain/           # brain dataset, outside Git
$DATA_ROOT/mountain/        # mountain dataset, outside Git
$DATA_ROOT/manifests/       # manifests/checksums
$SCRATCH_ROOT/runs/         # large outputs/checkpoints/logs, outside Git
```

Docker container mounts:

```text
/workspace/repo             # from $REPO_ROOT
/workspace/data             # from $DATA_ROOT, raw data read-only by default
/workspace/scratch          # from $SCRATCH_ROOT, outputs/checkpoints/logs
```

## Repository Layout

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── .rsync-exclude
├── configs/
│   ├── data/
│   ├── experiments/
│   ├── splits/
│   ├── docker/
│   └── public/
├── data/
│   ├── README.md
│   ├── manifests/
│   └── samples/                 # tiny synthetic/demo data only
├── docker/
│   ├── README.md
│   ├── dev-gpu/
│   │   ├── Dockerfile
│   │   └── compose.yaml
│   └── runtime/
├── docs/
│   ├── 00_index.md
│   ├── 01_project_brief.md
│   ├── 02_literature_review.md
│   ├── 03_methods.md
│   ├── 04_experiment_registry.md
│   ├── 05_results_summary.md
│   ├── 06_decision_log.md
│   ├── 07_todo.md
│   ├── 08_codex_workflow.md
│   ├── workflows/
│   ├── codex_prompts/
│   ├── todos/
│   └── reports/
├── experiments/
│   └── YYYY-MM-DD_short-slug/
├── logs/
├── notebooks/
├── outputs/                     # ignored except reviewed tiny summaries
├── papers/
├── private/                     # ignored; hostnames, local ops, secrets notes
├── scripts/
│   ├── data/
│   ├── experiments/
│   ├── ops/
│   ├── release/
│   └── reports/
├── src/
├── tests/
└── public/
    ├── README_REVIEW.md
    ├── ANONYMITY_AUDIT.md
    └── release_manifest.md
```

## Documentation Metadata Standard

Every durable Markdown document should start with front matter:

```yaml
---
doc_type: TODO_KIND
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: draft | active | archived | superseded
visibility: internal | public-review | public-final
public:
  initial_creation_date: YYYY-MM-DD | null
  first_upload_date: YYYY-MM-DD | null
  latest_public_update_date: YYYY-MM-DD | null
  anonymous_release_status: not_applicable | not_started | in_progress | audited | uploaded
related_code:
  - path/or/module.py
related_experiments:
  - experiments/YYYY-MM-DD_short-slug
---
```

Update `updated` whenever a document changes materially. Set public dates only when the public-review artifact is actually uploaded or updated.

## Dataset Rules

The real datasets live outside Git:

```text
$DATA_ROOT/brain/
$DATA_ROOT/mountain/
```

Rules:

- Do not commit raw dataset files.
- Do not modify raw data in place.
- Use read-only raw data mounts in Docker by default.
- Write processed/intermediate data to a versioned derived-data or scratch root, not into raw folders.
- Maintain dataset manifests in `data/manifests/` if small/review-safe or `$DATA_ROOT/manifests/` if private/large.
- Record data version, manifest path, split config, and preprocessing command in every experiment log.
- Define train/validation/test splits deterministically in `configs/splits/`.
- Avoid leakage by deciding the correct grouping unit before writing split code.

Fast manifest:

```bash
mkdir -p "$DATA_ROOT/manifests"
find "$BRAIN_DATA" -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort > "$DATA_ROOT/manifests/brain_manifest_$(hostname)_$(date +%F).tsv"
find "$MOUNTAIN_DATA" -type f -printf '%P\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS\n' | sort > "$DATA_ROOT/manifests/mountain_manifest_$(hostname)_$(date +%F).tsv"
```

## Multi-Server Model

Define server roles in `docs/00_index.md` or ignored private notes:

```text
PRIMARY_DEV_SERVER = TODO_HOST_A
GPU_WORKER_1       = TODO_HOST_B
GPU_WORKER_2       = TODO_HOST_C
DATA_SOURCE        = TODO_DATA_HOST
REGISTRY           = TODO_REGISTRY
```

Rules:

- Only `PRIMARY_DEV_SERVER` edits and commits source code by default.
- Worker servers run experiments from clean synced code states.
- Workers write outputs to scratch or ignored run folders.
- Promote a worker fix by creating a patch or branch and reviewing it on the primary repo.
- Do not let two servers independently edit the same branch and then use rsync to merge.

## Code Sync Strategy

Preferred hierarchy:

1. Use Git for source-of-truth changes.
2. Use one-way rsync to fast-mirror a clean working tree to workers.
3. Use dataset-specific rsync only for data/artifacts outside Git.
4. Never use bidirectional rsync to resolve source-code conflicts.

Dry-run code mirror:

```bash
rsync -azP --itemize-changes --delete \
  --exclude-from=.rsync-exclude \
  "$REPO_ROOT"/ TODO_GPU_WORKER:/home/TODO_USER/TODO_PROJECT_SLUG/ \
  --dry-run
```

Real mirror after approval:

```bash
rsync -azP --itemize-changes --delete \
  --exclude-from=.rsync-exclude \
  "$REPO_ROOT"/ TODO_GPU_WORKER:/home/TODO_USER/TODO_PROJECT_SLUG/
```

Recommended `.rsync-exclude` for code sync:

```text
.git/
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.env
.env.*
secrets/
private/
data/raw/
data/external/
data/derived/
outputs/runs/
outputs/tmp/
wandb/
mlruns/
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.tar
*.tar.gz
*.tar.zst
*.zip
*.log
docker/*.tar*
```

## Data Sync Examples

Dry-run brain transfer:

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

Use `DATASET=mountain` for the mountain dataset. Do not use `--delete` for raw datasets by default.

## Docker Workflow

Docker operations do not require the rsync impact briefing. Keep Docker commands reproducible and record image provenance for experiments.

Build/tag:

```bash
PROJECT=TODO_PROJECT_SLUG
REGISTRY=TODO_REGISTRY/TODO_PROJECT_SLUG
GIT_SHA=$(git rev-parse --short HEAD)
DATE=$(date +%Y%m%d)
IMAGE=${REGISTRY}:dev-gpu-${DATE}-${GIT_SHA}

docker build -f docker/dev-gpu/Dockerfile -t "$IMAGE" .
docker image inspect "$IMAGE" > outputs/inspection/docker_image_${DATE}_${GIT_SHA}.json
```

Run with separate repo/data/scratch mounts:

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

For long training jobs, prefer repo read-only:

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

Push/pull through registry:

```bash
docker push "$IMAGE"
ssh TODO_GPU_WORKER "docker pull '$IMAGE'"
```

Save/load without registry. The Docker save/load commands do not need the rsync briefing, but the rsync line does:

```bash
docker save "$IMAGE" | zstd -T0 -19 -o /tmp/${PROJECT}_${GIT_SHA}.tar.zst
rsync -azP /tmp/${PROJECT}_${GIT_SHA}.tar.zst TODO_GPU_WORKER:/tmp/
ssh TODO_GPU_WORKER "zstd -dc /tmp/${PROJECT}_${GIT_SHA}.tar.zst | docker load"
```

Do not commit Docker image archives.

## Experiment Workflow

Before launching a real run:

1. Update `docs/04_experiment_registry.md`.
2. Create `experiments/YYYY-MM-DD_short-slug/`.
3. Add `codex_execution_spec.md` if Codex will implement or modify code for the run.
4. Add `technical_log.md` with known limitations near the top.
5. Confirm dataset manifest and split config.
6. Confirm Docker image tag or digest.
7. Run the smallest smoke command.
8. Launch the real run only after approval if in plan mode.

Required `technical_log.md` fields:

```text
Run ID:
Date started:
Date updated:
Status:
Owner/agent:
Server:
GPU model/count:
Git commit:
Branch:
Docker image tag/digest:
Dataset roots:
Dataset manifests:
Split config:
Preprocessing command:
Training/evaluation command:
Hyperparameters/config:
Metrics:
Artifact paths:
Known limitations:
Next decision:
```

## Worker Lanes

Use lightweight lanes so multiple humans/agents can work in parallel:

- `literature`
- `methods`
- `experiments`
- `docker`
- `server-sync`
- `paper-writing`
- `anonymous-release`
- `brainstorming`

Each lane should maintain a todo file under `docs/todos/`.

## Anonymous Review Code Strategy

Build the review-safe codebase continuously, not at the end.

Anonymous-safe by default:

```text
src/
configs/public/
scripts/experiments/
scripts/reports/
tests/
public docs
```

Private or ignored:

```text
private/
docs/private/
server-specific launch files
personal automation
credentials
hostnames
institution identifiers
absolute local paths
```

Recommended public-review layout:

```text
public/
├── README_REVIEW.md
├── ANONYMITY_AUDIT.md
├── release_manifest.md
├── environment.md
├── expected_runtime.md
├── tiny_demo_commands.md
└── known_limitations.md
```

## Commit Safety Checks

For commit preparation, inspect:

```bash
git status --short
git diff --stat
git diff --cached --stat
git ls-files -o --exclude-standard | head -200
git submodule status || true
find . -type f -size +10M \
  -not -path './.git/*' \
  -not -path './private/*' \
  -not -path './outputs/*' \
  -not -path './data/*' \
  | sort
```

Never commit raw data, private datasets, checkpoints, model weights, large outputs, cache folders, W&B/MLflow runs, Docker image tarballs, credentials, private hostnames, absolute personal paths, or review-breaking identity metadata. Prefer pathspec-limited staging and avoid `git add .`.

## Coding Style

- Python 3.10+ unless Docker standardizes another version.
- Use type hints for public functions where practical.
- Prefer CLI scripts with `--help`.
- Use config files for experiment parameters.
- Prefer deterministic seeds and recorded random states.
- Add smoke tests or synthetic tests for parsers, data loaders, losses, metrics, and release scripts.
- Keep notebooks exploratory; production logic belongs in `src/` or `scripts/`.

## Current Commands Template

```bash
python -m pytest -q
python scripts/data/inspect_brain_dataset.py --help
python scripts/data/inspect_mountain_dataset.py --help
python scripts/experiments/run_smoke.py --help
python scripts/experiments/train.py --help
python scripts/experiments/evaluate.py --help
python scripts/reports/build_technical_report.py --help
python scripts/release/build_anonymous_release.py --help
```

Docker smoke:

```bash
docker run --rm --gpus all "$IMAGE" python -m pytest -q
```

## Codex Bootstrap Prompt

```text
[adapt project]
This is a new multi-server GPU research repo. Keep Git/GitHub as the source of truth for code/docs/configs/tests. Keep raw brain and mountain datasets outside Git under dedicated per-server DATA_ROOT folders. Docker must mount the repo and data as separate volumes: /workspace/repo and /workspace/data, with raw data read-only by default, plus /workspace/scratch for outputs. Add a safety gate only for rsync: every rsync must explain source, target, direction, delete behavior, exclusions, dry-run, real command, and log/manifest plan. Docker operations do not require the rsync-style impact explanation. No bidirectional source-code sync; no raw data in the repo. Build anonymous-review-safe code continuously. Return a plan before editing.
```

## Adaptation Notes From Previous Project

Keep from the old project pattern:

- Plan mode and approval before destructive actions.
- Commit preparation and summarize-and-commit workflows.
- Experiment registry before major runs.
- Technical logs for every real experiment.
- Documentation updates after meaningful work.
- Commit safety checks for data, checkpoints, outputs, credentials, Docker artifacts, and large files.
- Smoke tests before long runs.

Adapt or remove:

- Medical-physics-specific constraints.
- DoseRAD-specific commands and Docker image names.
- Dataset-specific leakage rules that do not apply to brain/mountain.
- Any old project hostnames, paths, usernames, W&B entities, image tags, and paper notes.

Add for this project:

- Explicit brain/mountain dataset roots and manifests.
- Multi-server single-writer model.
- One-way rsync code/data workflows with mandatory rsync-only impact briefing.
- Docker repo/data/scratch mount workflow.
- Public-review/anonymous-release lane from day one.
- Metadata front matter with created, updated, public initial creation, public upload, and public update dates.
