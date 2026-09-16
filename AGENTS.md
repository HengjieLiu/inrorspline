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
  repo_source_of_truth: "git/GitHub for source code, docs, configs, tests, and source-only review material",
  artifact_source_of_truth: "${ARTIFACT_ROOT} for generated experiment, review, and migrated project artifacts; never Git",
  code_write_model: "single-writer primary development repo",
  worker_model: "GPU workers are execution mirrors unless explicitly promoted",
  datasets: ["brain", "lungct"],
  data_source_of_truth: "dataset-specific server-local roots plus ARTIFACT_ROOT for project-generated data; never Git",
  docker_model: "container gets separate repo/raw-data/artifact mounts",
  repo_mount_in_container: "/workspace/repo",
  data_mount_in_container: "/workspace/data",
  artifact_mount_in_container: "/database/inrorspline",
  artifact_env_in_container: "ARTIFACT_ROOT=/database/inrorspline",
  raw_data_mode: "read-only by default",
  rsync_policy: "explain impact first, dry-run first, real sync only after approval",
  docker_policy: "normal requested operation; no rsync-style impact explanation required",
  artifact_policy: "source-only Git worktree; generated artifacts and executed notebooks are external-only",
  public_review_policy: "anonymous-safe code and docs from day one"
}
```

## Command Interface

Use these bracket commands in prompts:

```text
[use plan mode]              # do not edit files, stop jobs, launch jobs, remove artifacts, or run destructive commands. Convert the request into a concrete plan, summarize assumptions and risks, recommend defaults, list files likely to change, identify verification steps, ask concise clarifying questions if needed, suggest recommendations if any, and wait for explicit approval before acting. This command should not be superseded.
[adapt and plan]             # reuse rules of [use plan mode], but more specifically understand the project context and adapt the current plan files into a new concrete plan, summarize key adaptation needed, ask a series of clarifying questions and if appropriate provide selection answers, suggest recommendations if any, and wait for explicit approval before acting. This command should not be superseded.
[adapt project]              # adapt instructions/docs when assumptions change; plan first
[adapt AGENTS.md]            # revise AGENTS.md after classifying legacy rules
[agents sync]                # propose AGENTS.md updates from current repo/conversation context
[sync-code plan]             # one-way primary repo -> worker rsync; explain + dry-run first
[sync-data plan]             # one-way DATA_ROOT -> DATA_ROOT rsync; explain + dry-run first
[artifact-store migration plan] # copy/check/retain migration to ARTIFACT_ROOT; never delete source without separate approval
[rsync explain]              # required before every rsync command
[docker run]                 # Docker build/run commands; no rsync-style impact explanation needed
[docker image transfer]      # registry push/pull, or save/load; rsync step follows rsync rule
[experiment launch plan]     # plan a real run with registry/log/provenance first
[anonymous release audit]    # audit review-facing code/docs for identity leaks
[add git external] <url>     # add a GitHub repo as an external/ submodule and commit
[prepare commit]             # summarize status and propose commit commands; do not stage/commit
[summarize and commit]       # produce commit plan first; wait for approval
[summarize and commit <path>] # same as above, scoped to the requested path(s)
[summarize and commit by groups] # propose a topic-clustered commit series; wait for approval
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
- Large generated-artifact handling. Inspect untracked files with sizes before
  deciding what to exclude, for example with
  `git ls-files --others --exclude-standard -z | xargs -0 du -h | sort -h`.
  Do not exclude or ignore files solely because they are CSV, JSON, notebook, or
  image files; small review tables, summaries, figures, and paper artifacts may
  belong in Git. When a large or transient generated artifact should remain out
  of Git and should stop appearing in future commit audits, propose an
  exact-path or tightly scoped `.gitignore` entry with a short comment naming
  the approximate size, purpose, and tracked summary/report that replaces it.
  Use `.git/info/exclude` instead of committed `.gitignore` for host-local
  clutter that should not become repo policy. If the artifact is already
  tracked, do not remove it from Git without explicit approval for
  `git rm --cached`.
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

When the worktree should be divided into multiple independently reviewable
commits, use `[summarize and commit by groups]` instead.

### `[summarize and commit <path>]`

Use when the user wants a commit plan scoped to one or more requested paths.

Codex must scope the audit, staging plan, and commit message to the requested
path or paths only. Do not stage changes outside the requested scope unless
required metadata is needed for that scope, such as `.gitmodules` for a
submodule change. Explain any such exception clearly in the plan.

### `[summarize and commit by groups]`

Use when the worktree contains multiple independently reviewable topics that
should be committed separately. This command is approval-gated.

Codex must not stage or commit immediately.

First ground the audit in the repository:

- Read `AGENTS.md`, every Markdown file under `AGENTS/` if that directory
  exists, and README files relevant to the changed paths.
- Inspect Git status, staged and unstaged changes, ignored files, submodule
  state, and the size and type of untracked artifacts.
- Treat existing dirty-tree changes as user work unless they were clearly
  produced by the current task.
- Identify dependencies between code, configuration, tests, documentation,
  generated review artifacts, and provenance records.

Group changes by topic, ownership boundary, dependency order, and
reviewability. Prefer commits that are independently understandable and
verifiable.

For each proposed commit, provide:

- A concise subject line.
- A comprehensive but compact message body.
- Exact paths to stage.
- Files and folders intentionally excluded.
- Paths that should remain outside Git or receive a tightly scoped
  `.gitignore` or `.git/info/exclude` entry.
- Verification to run before staging.
- Verification to run against the staged diff.
- Any dependency on an earlier proposed commit.

Rules:

- Use explicit path staging only. Never use `git add .` or `git add -A`.
- Do not stage unrelated dirty-tree changes.
- Do not commit runtime outputs, logs, predictions, checkpoints, model
  weights, medical-image volumes, Python caches, notebook checkpoints,
  credentials, private host metadata, or large/transient generated artifacts.
- Small review tables, summaries, notebooks, and figures may be included when
  they are intentional repository or paper artifacts and pass the normal size,
  provenance, and anonymity audit.
- Avoid splitting one file across commits by default. If a file genuinely
  contains changes for multiple groups, assign it to the most appropriate
  group or propose an explicit hunk-staging plan for approval.
- Order commits so prerequisites such as code, configuration, and tests land
  before dependent documentation or synchronized summaries.
- Refresh repo-local experiment summaries only at the correct point in the
  sequence so `repo_commit`, source hashes, and provenance fields describe the
  intended committed code/configuration state.
- Do not push unless explicitly requested.

After presenting the complete series, stop and wait for explicit approval. The
user may approve all groups, edit the grouping, or approve selected groups.

After approval, process one group at a time:

1. Recheck the worktree for drift.
2. Run the planned pre-stage verification.
3. Stage only the approved explicit paths.
4. Inspect `git diff --cached --check`, `git diff --cached --stat`, and
   `git diff --cached --submodule`.
5. Confirm no unrelated paths entered the index.
6. Commit with the approved message.
7. Record the commit hash and recheck status before starting the next group.

If the worktree, dependencies, or verification results differ materially from
the approved plan, stop and present a revised plan before continuing.

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
- Log meaningful real syncs outside the repo under
  `${ARTIFACT_ROOT}/migration_manifests/`.
- Artifact-store migrations are copy/check/retain operations. Do not use
  `--delete`, `--remove-source-files`, or `--inplace` for an initial or final
  migration pass. Preserve source trees for at least 30 days after validation;
  rerun checksum comparison and obtain separate approval before any deletion.
- For an artifact-store migration, validate a small metadata canary first, use
  `-aHAXS -x` with ownership/group preservation disabled, freeze writers for a
  final delta pass, and require a checksum dry-run plus count/size/mtime
  inventories before declaring the destination valid. Treat ACL, xattr, mode,
  path, file-type, or modification-time preservation failures as blockers.

## Path and Mount Model

```bash
export PROJECT_SLUG=inrorspline
export REPO_ROOT="$HOME/code_sync/$PROJECT_SLUG"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:?set ARTIFACT_ROOT to the external artifact store}"
export ARTIFACT_RUN_ROOT="$ARTIFACT_ROOT/runs"
export ARTIFACT_REVIEW_ROOT="$ARTIFACT_ROOT/repo_experiments"
export ARTIFACT_MANIFEST_ROOT="$ARTIFACT_ROOT/migration_manifests"
export DATA_ROOT="${DATA_ROOT:?set the raw-data root in ignored local configuration}"
export BRAIN_DATA="${BRAIN_DATA:?set the brain-data root in ignored local configuration}"
export LUNGCT_DATA="${LUNGCT_DATA:?set the LungCT-data root in ignored local configuration}"
export LUNGCT_PTVREG_ISO1MM="${LUNGCT_PTVREG_ISO1MM:?set the pTVReg root in ignored local configuration}"
```

Expected host layout:

```text
$REPO_ROOT/                             # Git-managed source only
$ARTIFACT_ROOT/runs/                    # checkpoints, fields, run logs, outputs
$ARTIFACT_ROOT/{ptvreg,prepared,dual_author_data,logs}/ # migrated project data
$ARTIFACT_ROOT/repo_experiments/        # figures, tables, reports, executed notebooks
$ARTIFACT_ROOT/migration_manifests/     # inventories, checksums, transfer logs
external raw dataset roots               # host-local, outside Git and ARTIFACT_ROOT when retained
```

`ARTIFACT_ROOT` is host-specific operational context. Committed code and docs
must use the variable only; the actual mount path belongs in ignored
`private/data_sources.md`. The Git worktree is source-only: generated figures,
PDFs, CSV/JSON result tables, reports, live-monitor files, checkpoints, dense
fields, and executed notebook copies all belong under `ARTIFACT_ROOT`, even
when they are small. Keep the narrow `.gitignore` policy so accidental
repo-local artifacts remain visible for review rather than being silently
hidden.

Expected container layout:

```text
/workspace/repo             # mounted from $REPO_ROOT
/workspace/data             # optional raw-data mounts; read-only by default
/database/inrorspline       # compatibility bind of $ARTIFACT_ROOT for legacy launchers
ARTIFACT_ROOT=/database/inrorspline  # required container environment for artifact resolvers
```

## Docker Execution Default

Use the running Docker container below by default for preprocessing and
data-inspection Python work, unless the user explicitly says to use a different
environment:

```text
legacy_container_name: py310sitreg_v1_test1
default_container_name: py310sitreg_v1_artifact_store
default_container_image: hengjieliu96/dl_dir:py310sitreg_v1
```

Default interactive entry:

```bash
docker exec -it py310sitreg_v1_artifact_store bash
```

Default direct Python entry:

```bash
docker exec py310sitreg_v1_artifact_store python <script-and-args>
```

The default container has a verified `ARTIFACT_ROOT=/database/inrorspline`
environment and compatible `ARTIFACT_ROOT:/database/inrorspline:rw` bind. The
legacy container remains available only as a rollback reference; do not stop
or repurpose it without explicit approval.

Inside this container, use the legacy path mapping:

```text
/home/hengjie  -> /homebase
/data/hengjie  -> /database
```

All generated `inrorspline` experiment material should default to:

```text
host:   ${ARTIFACT_ROOT}
docker: /database/inrorspline
```

The container must receive a direct read-write bind from `ARTIFACT_ROOT` to
`/database/inrorspline`, while any existing raw-data bind at `/database` may
remain in place. Small review figures, CSV summaries, reports, and executed
notebooks are generated artifacts too; retain only their source code,
specifications, and unexecuted notebook sources in the repo.

Do not use the native Python environment for preprocessing, mask generation,
image-loading audits, notebooks, or dependency-sensitive data work unless the
user explicitly requests it. Native shell commands remain allowed for repo
inspection, `rg`, `sed`, `git status`, Docker status checks, and other
non-Python/non-preprocessing tasks.

## Notebook Review Habit

When the user asks for a notebook, especially for data inspection, visual audit,
or qualitative review, keep an output-free notebook source in the repo and use
an executed copy under `ARTIFACT_ROOT/repo_experiments/<experiment-key>/` as the
primary review surface. That executed copy must display important figures,
overlays, tables, and summaries inline. Companion scripts may generate the
external outputs, but must not write them back into the worktree.

After creating or materially revising a review notebook, execute the external
copy in the intended environment unless the user explicitly asks not to run it
or the run would be unsafe or prohibitively expensive. For preprocessing or
data-inspection notebooks, use the default Docker container. Before any source
notebook is committed, clear all cell outputs; never treat an executed notebook
or embedded figure output as a Git artifact.

Run `python -B scripts/experiments/check_tracked_notebook_outputs.py` before
declaring the worktree source-only or preparing a commit. It must report no
Git-tracked notebook cell outputs.

## Local Data Source Map

Host-specific raw-data and legacy-workspace paths are private operational context. They must live in ignored files such as `private/data_sources.md`, not in public configs or review-facing docs.

Before OASIS/brainMR or LungCT/4DCT data work, check `private/data_sources.md` if it exists. It should summarize:

- legacy task roots to inspect for loaders, reports, and protocol state
- host and container path mappings
- the configured external `ARTIFACT_ROOT` and its `/database/inrorspline`
  compatibility bind
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
7. Configure `ARTIFACT_ROOT`; direct run, review, monitor, and executed-notebook
   outputs outside the repo and record symbolic roots in the execution spec.
8. Confirm Docker image tag or digest.
9. Run the smallest relevant smoke command.
10. Launch the real run only after approval if in plan mode.

Do not commit raw data, checkpoints, generated outputs (including figures,
tables, reports, or notebook outputs), Docker image archives, private hostnames,
credentials, or identity-breaking metadata.

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
