---
doc_type: artifact_store_migration
status: complete
---

# External Artifact Store Migration

## Status

Completed 2026-09-03. The external store passed destination access and
metadata-canary checks, a writer-free final delta, checksum dry-run comparisons,
and metadata inventories. The Git worktree cleanup completed only after those
checks passed. This record is symbolic by design: the literal host mapping is
private operational context in `private/data_sources.md`.

The former data source tree is intact and rollback-only through **2026-10-03**.
No source-data subtree may be deleted before that date without a fresh checksum
comparison and separate explicit approval for the named subtree.

## Target Layout

The host-specific destination is represented in committed material only as
`${ARTIFACT_ROOT}`. Its private host mapping belongs in
`private/data_sources.md`.

```text
${ARTIFACT_ROOT}/
├── runs/                         # checkpoints, dense fields, run outputs
├── ptvreg/                       # migrated project artifacts
├── prepared/                     # migrated prepared data
├── dual_author_data/             # migrated project data
├── logs/                         # durable run logs
├── repo_experiments/             # generated review material and executed notebooks
└── migration_manifests/          # inventories, checksums, and transfer logs
```

The external store must be mounted at `/database/inrorspline` in legacy
containers so existing run-root paths remain valid. Raw-data mounts may remain
separate and read-only.

Repository-generated material will be archived under
`${ARTIFACT_ROOT}/repo_experiments/migration_<YYYY-MM-DD>/`, preserving paths
relative to the repository for material originating in `experiments/`,
`outputs/`, and output-bearing notebooks. The worktree cleanup itself is
performed only after the corresponding external copy validates.

## Migration Rules

- Copy source data and repository-generated material first, preserving relative
  paths beneath their respective external destinations.
- Validate a metadata canary, initial copy, writer-free final delta, checksum
  comparison, and count/size/mtime inventories before accepting a subtree.
- Store all migration manifests outside Git under
  `${ARTIFACT_ROOT}/migration_manifests/`.
- Do not use destructive rsync options during copy or validation. Keep each
  validated source subtree unused for at least 30 days.
- Before any later deletion, rerun the checksum comparison and obtain explicit
  per-subtree approval. Deletion is not authorized by this migration record.

## Repository Policy After Validation

The Git worktree will remain source-only. It may contain code, configuration,
documentation, hand-authored execution specifications/logs, and output-free
notebook sources. Generated figures, PDFs, result tables, reports, monitor
files, checkpoints, dense fields, and executed notebooks belong only in the
external artifact store.

Historical registry rows deliberately keep the paths they recorded at execution
time. They should not be mass-rewritten; a symbolic migration manifest is the
authoritative link between a verified source subtree and its external
destination.

## Verification Record

- Data scopes verified: `runs`, `ptvreg`, `prepared`, `dual_author_data`, and
  `logs` — 100,385 entries and 935,539,955,548 regular-file bytes in total.
- Repository archive scopes verified: `experiments/`, `outputs/`,
  `notebooks/`, and the four scoped documentation PDFs.
- The final checksum-based rsync dry run reported zero created, deleted, or
  transferred files in every data and repository scope.
- Final metadata inventories matched paths, types, modes, modification times,
  regular-file sizes, symlink targets, and regular-file hard-link counts.
  Ownership/group, ctime, atime, birth time, and filesystem-internal directory
  allocation size were intentionally not treated as portable acceptance fields.
- The source-only cleanup removed 4,158 verified generated worktree files
  (3,110,054,416 bytes) without modifying Git history or source data. Nineteen
  Git-tracked notebooks had embedded outputs cleared; their executed copies
  remain in the archive. Authored source configurations and hand-authored
  specifications remain in Git.
- A replacement container has `ARTIFACT_ROOT=/database/inrorspline`, retains
  the legacy raw-data bind at `/database`, and has a verified writable nested
  artifact-store mount at `/database/inrorspline`. The legacy container remains
  untouched as a rollback reference.

Detailed immutable operational logs, checksum output, inventories, cleanup
decisions, and execution records are stored beneath
`${ARTIFACT_ROOT}/migration_manifests/2026-09-02_external_artifact_store_v1/`.

## Retention and Future Operation

The 30-day retention period is active. New generated output must resolve below
`${ARTIFACT_ROOT}/runs/` or `${ARTIFACT_ROOT}/repo_experiments/`; accidental
repo-local output remains visible for rejection rather than being broadly
ignored. The old source tree is not a future output destination.
