# AGENTS.md - inrorspline Research Repo Instructions

## Start Here

Before coding, syncing, dataset movement, Docker work, paper writing, or experiment execution, read:

- `docs/00_index.md` for the project map and current status.
- `docs/01_project_brief.md` for the scientific goal and expected claims.
- `docs/03_methods.md` for method and protocol definitions.
- `docs/implementation/README.md` for code-derived implementation summaries
  used in method review and paper writing.
- `docs/04_experiment_registry.md` before planning or launching real runs.
- `docs/05_results_summary.md` before writing conclusions.
- `docs/06_decision_log.md` before changing major design choices.
- `docs/standards/README.md` and the relevant dataset standard before planning,
  launching, or interpreting experiments.
- `docs/08_codex_workflow.md` for repo-specific Codex conventions.
- `docs/workflows/server_sync_workflow.md` before using `rsync`.
- `docs/workflows/docker_gpu_workflow.md` before Docker changes or launches.
- `docs/workflows/anonymous_release_workflow.md` before touching review-facing files.
- `docs/reports/protocol_warnings/` before interpreting LungCT/4DCT comparisons.

## Project Goal

This repository supports a reproducible study of continuous deformation models for medical image registration. The core question is whether observed gains come from the INR label itself, spline/local-control structure, optimization path, regularization, or deformation-regime fit.

Primary dataset handles:

- `brain`: OASIS brain registration protocol, currently centered on the seed-3 shuffled first10 plan.
- `lungct`: DIR-Lab 4DCT/COPD-style LungCT protocols, with careful protocol tags for native IDIR, pTVReg-style crops, and DirectCP/SINR variants.

## Core Operating Object

```text
PROJECT_OBJECT = {
  phase: "new_project_scaffold",
  project_slug: "inrorspline",
  repo_source_of_truth: "git/GitHub for code, docs, configs, tests, public-review artifacts",
  code_write_model: "single-writer primary development repo",
  worker_model: "GPU workers are execution mirrors unless explicitly promoted",
  datasets: ["brain", "lungct"],
  data_source_of_truth: "server-local DATA_ROOT folders, never Git",
  docker_model: "container gets separate repo/data/scratch mounts",
  repo_mount_in_container: "/workspace/repo",
  data_mount_in_container: "/workspace/data",
  scratch_mount_in_container: "/workspace/scratch",
  raw_data_mode: "read-only by default",
  rsync_policy: "explain impact first, dry-run first, real sync only after approval",
  docker_policy: "normal requested operation; no rsync-style impact explanation required",
  public_review_policy: "anonymous-safe code and docs from day one"
}
```

## Command Interface

Use these bracket commands in prompts:

```text
[use plan mode]              # do not edit files, stop jobs, launch jobs, remove artifacts, or run destructive commands. Convert the request into a concrete plan, summarize assumptions and risks, recommend defaults, list files likely to change, identify verification steps, ask concise clarifying questions if needed, suggest recommendations if any, and wait for explicit approval before acting. This command should not be superceded.
[adapt project]              # adapt instructions/docs when assumptions change; plan first
[adapt AGENTS.md]            # revise AGENTS.md after classifying legacy rules
[agents sync]                # propose AGENTS.md updates from current repo/conversation context
[sync-code plan]             # one-way primary repo -> worker rsync; explain + dry-run first
[sync-data plan]             # one-way DATA_ROOT -> DATA_ROOT rsync; explain + dry-run first
[rsync explain]              # required before every rsync command
[docker run]                 # Docker build/run commands; no rsync-style impact explanation needed
[docker image transfer]      # registry push/pull, or save/load; rsync step follows rsync rule
[experiment launch plan]     # plan a real run with registry/log/provenance first
[anonymous release audit]    # audit review-facing code/docs for identity leaks
[add git external] <url>     # add a GitHub repo as an external/ submodule and commit
[prepare commit]             # summarize status and propose commit commands; do not stage/commit
[summarize and commit]       # produce commit plan first; wait for approval
[summarize and commit <path>] # same as above, scoped to the requested path(s)
```

### `[agents sync]`

Use when the current conversation reveals a missing, stale, ambiguous, or non-standard repo instruction that should become durable guidance in `AGENTS.md`.

Codex must:

- Review the recent conversation and relevant repo context.
- Propose `AGENTS.md` changes first unless the user explicitly asks to update immediately.
- Classify each proposed change as `ADD`, `UPDATE`, `REMOVE`, or `CLARIFY`.
- Keep changes scoped to durable operating rules, command conventions, safety gates, protocol warnings, and workflow standards.
- Avoid encoding one-off task details, private host paths, temporary experiment notes, or conclusions that belong in reports instead.
- After editing, summarize the changed rule and show verification status.

Prompt:

```text
[agents sync]
Review the current conversation and repo context for missing, stale, ambiguous, or non-standard instructions that should be captured in AGENTS.md. Propose changes first. Classify each proposed change as ADD, UPDATE, REMOVE, or CLARIFY. Do not edit until approved unless I explicitly say to update now.
```

### `[add git external] <github-url>`

Use when the user provides a GitHub repository URL that should be tracked as an
external dependency/source reference.

Codex must:

- Require an explicit GitHub URL in the prompt.
- Derive the target path as `external/<repo-name>` unless the user gives a
  different path.
- Add the repo with `git submodule add <github-url> external/<repo-name>`.
- Treat the external repo as a submodule gitlink, not vendored source copied
  into this repository.
- Inspect the submodule state with `git submodule status` and `git status`.
- Stage only `.gitmodules` and the new `external/<repo-name>` gitlink.
- Commit only the submodule addition, using a message like
  `Add external submodule <repo-name>`.
- Do not stage or commit unrelated worktree changes.
- If `.gitmodules` or the target `external/<repo-name>` already has local
  changes, stop and ask before proceeding.

Prompt:

```text
[add git external] https://github.com/OWNER/REPO
Add this repository under external/ as a Git submodule and commit only that
submodule addition.
```

### `[summarize and commit]`

Use when the user wants Codex to prepare a commit from the current worktree.
This command is approval-gated.

Codex must not commit immediately. First inspect the working tree and return a
commit plan for approval. Include:

- A concise topic-grouped change summary.
- Folders with new files planned for commit.
- Files or folders intentionally excluded.
- Submodule status and planned handling.
- Verification already run or recommended.
- One comprehensive but concise commit message. "Concise" means no filler, not
  vague. The subject should identify the concrete deliverable and the main
  protocol/behavioral change, for example `Add OASIS first10 BE sweep runner and
  population ARC aggregation`, not `Update experiments`. For multi-topic
  commits, propose a subject plus a short body with 2-5 bullets naming the
  actual changes: new runners/aggregators, protocol or metric conventions,
  generated review artifacts, gated launch behavior, and verification. For
  correction work, mention what was fixed, such as orientation, foreground-mask
  convention, Docker/notebook execution, verification, or data-safety behavior.
- The exact `git add` and `git commit` commands that would run after approval.

After presenting the plan, wait for explicit user approval before staging or
committing. If approved, stage only the planned files and commit with the
proposed or user-edited message.

### `[summarize and commit <path>]`

Use when the user wants a commit plan scoped to one or more requested paths.

Codex must scope the audit, staging plan, and commit message to the requested
path or paths only. Do not stage changes outside the requested scope unless
required metadata is needed for that scope, such as `.gitmodules` for a
submodule change. Explain any such exception clearly in the plan.

## Rsync-Only Safety Gate

Only `rsync` requires a mandatory impact explanation. Docker operations do not require this special briefing.

Before any rsync command, provide:

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

Rules:

- Always dry-run first for dataset rsync or any rsync with `--delete`.
- Never use bidirectional rsync for source code.
- Never use `--delete` on raw datasets by default.
- Never sync raw data into the Git repo.
- Log meaningful real syncs in `logs/YYYY-MM.md` or `outputs/inspection/`.

## Path and Mount Model

```bash
export PROJECT_SLUG=inrorspline
export REPO_ROOT="$HOME/code_sync/$PROJECT_SLUG"
export DATA_ROOT="/data/$USER/$PROJECT_SLUG"
export SCRATCH_ROOT="/scratch/$USER/$PROJECT_SLUG"
export BRAIN_DATA="$DATA_ROOT/brain"
export LUNGCT_DATA="$DATA_ROOT/lungct"
```

Expected host layout:

```text
$REPO_ROOT/                 # Git-managed code/docs/configs/tests
$DATA_ROOT/brain/           # OASIS and related brain data, outside Git
$DATA_ROOT/lungct/          # LungCT/4DCT/COPD data, outside Git
$DATA_ROOT/manifests/       # manifests/checksums for data integrity
$SCRATCH_ROOT/runs/         # large experiment outputs, outside Git
```

Expected container layout:

```text
/workspace/repo             # mounted from $REPO_ROOT
/workspace/data             # mounted from $DATA_ROOT; raw data read-only by default
/workspace/scratch          # mounted from $SCRATCH_ROOT; outputs/checkpoints/logs
```

## Docker Execution Default

Use the running Docker container below by default for preprocessing and
data-inspection Python work, unless the user explicitly says to use a different
environment:

```text
default_container_name: py310sitreg_v1_test1
default_container_image: hengjieliu96/dl_dir:py310sitreg_v1
observed_container_id: c525c6217286
```

Default interactive entry:

```bash
docker exec -it py310sitreg_v1_test1 bash
```

Default direct Python entry:

```bash
docker exec py310sitreg_v1_test1 python <script-and-args>
```

Inside this container, use the legacy path mapping:

```text
/home/hengjie  -> /homebase
/data/hengjie  -> /database
```

Large `inrorspline` experiment artifacts, checkpoints, dense fields, and
per-run logs should default to:

```text
host:   /data/hengjie/inrorspline
docker: /database/inrorspline
```

Only large data and run artifacts belong there. Small review figures, CSV
summaries, and reports may stay under the repo/homebase when they are intended
for immediate inspection or later documentation.

Do not use the native Python environment for preprocessing, mask generation,
image-loading audits, notebooks, or dependency-sensitive data work unless the
user explicitly requests it. Native shell commands remain allowed for repo
inspection, `rg`, `sed`, `git status`, Docker status checks, and other
non-Python/non-preprocessing tasks.

## Notebook Review Habit

When the user asks for a notebook, especially for data inspection, visual audit,
or qualitative review, design the notebook as the primary review surface. It
must display the important figures, overlays, tables, and summaries inline in
the notebook, not only save files to disk. Companion scripts are welcome for
reproducibility, but the notebook should call them and then render the generated
outputs directly so the user can inspect results without leaving Jupyter.

Notebook outputs can be cleared before commit unless the notebook is intended as
a small public report. The cells needed to regenerate and display the review
figures should remain in the notebook.

After creating or materially revising a review notebook, run it in the intended
environment so the user can open it and inspect populated outputs directly,
unless the user explicitly asks not to run it or the run would be unsafe or
prohibitively expensive. For preprocessing/data-inspection notebooks, execute
inside the default Docker container and keep the executed notebook outputs when
the user needs immediate visual review.

## Local Data Source Map

Host-specific raw-data and legacy-workspace paths are private operational context. They must live in ignored files such as `private/data_sources.md`, not in public configs or review-facing docs.

Before OASIS/brainMR or LungCT/4DCT data work, check `private/data_sources.md` if it exists. It should summarize:

- legacy task roots to inspect for loaders, reports, and protocol state
- host and container path mappings
- raw OASIS/brainMR image, segmentation, split, and baseline-displacement roots
- raw LungCT/4DCT image, landmark, lung-mask, and pTV-preprocessed output roots
- protocol warnings from `docs/reports/protocol_warnings/`

Do not copy private absolute paths from `private/data_sources.md` into `README.md`, `configs/public/`, `public/`, or anonymous-release docs. For public release, replace them with environment variables and dataset-layout descriptions.

## Experiment Rules

Before launching a real run:

1. Update `docs/04_experiment_registry.md`.
2. Review `docs/standards/README.md`,
   `docs/standards/fair_experiment_checklist.md`, and the relevant
   dataset-specific standard.
3. Create `experiments/YYYY-MM-DD_short-slug/`.
4. Add `codex_execution_spec.md` if Codex will implement or modify code for the run.
5. Add `technical_log.md` with known limitations near the top.
6. Confirm dataset manifest and split config.
7. Confirm Docker image tag or digest.
8. Run the smallest relevant smoke command.
9. Launch the real run only after approval if in plan mode.

Do not commit raw data, checkpoints, generated outputs, Docker image archives, private hostnames, credentials, or identity-breaking metadata.

## LungCT Interpretation Rule

Do not treat original IDIR, pTVReg, SINR, and DirectCP LungCT rows as clean method-only comparisons unless preprocessing, crop, mask, metric, interpolation, regularization scale, and evaluation convention are matched or explicitly tagged. Read `docs/reports/protocol_warnings/` first.

For native/full-volume LungCT IDIR-style runs, the lung-mask optimization
convention is fixed/T00 mask only. Moving/T50 masks may exist for QA or paired
mask audits, but they are not part of the original/native IDIR optimization
domain unless a run is explicitly tagged as a non-standard variant.

## OASIS BrainMR Split And Foreground Rule

For short OASIS/brainMR tests, use the seed-3 shuffled `pair_test_200`
protocol documented in `docs/standards/oasis_brainmr_protocol.md`. Do not pick
the first 10 rows directly from sorted `pair_test_200.txt`.

For OASIS/brainMR foreground masks, default to raw image-positive foreground:
`image > 0`. Do not use label foreground `seg > 0` as the default foreground,
sampling-domain, or foreground-scoped analysis mask. `seg > 0` is allowed for
label metrics and visual/comparison audits, or as an explicit documented
protocol override.
