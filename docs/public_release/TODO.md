# Release checklist

Human checklist for `release_spec.json`. Completion here refers to release
work, not whether the underlying historical experiment finished.

Publication gate (2026-09-30): **SINR permission is pending**; historical
implementation checkmarks below do not authorize publishing this candidate.

## Audit completed

- [x] Read attached discussion, repository rules, paper PDF, source, and results.
- [x] Record first-release reproduction scope and COPD extension choice.
- [x] Record updated-main-table choice and frozen external-reference policy.
- [x] Identify three clean gitlink pins and wrapper migration requirements.
- [x] Locate canonical Figure 2 data and verify the public PNG hash.
- [x] Find September 16 and September 24 OASIS update candidates.
- [x] Resolve Q04: September 16 paired timing plus separate historical OASIS100
  accuracy; September 24 CP-BE is not selected.
- [x] Identify 4DCT/COPD crop mismatch and incompatible dependency declarations.
- [x] Create private human/machine release-planning records.

## P0 — Freeze the scientific contract

- [x] R01: Freeze the selected updated-table source, cohort, schedule, timing
  boundaries, and provenance. Keep historical paper Table 1 separately.
- [x] R02: Export ordered OASIS100 and applicable timing pair manifests, verify
  against frozen source rows, and hash them; keep N=10/N=100 labels explicit.
- [x] R03: Select COPD rows with verified per-case coverage and source hashes;
  label as post-paper extension. Keep Lung acceleration pending.
- [x] R04 / CROP-001: Resolve the legacy one-based/zero-based crop
  discrepancy through explicit corrected/legacy preprocessing modes and distinct
  protocol IDs. Preserve historical COPD inputs/results, plan a separately
  versioned corrected COPD path, and fix contradictory TRE-standard and mask
  metadata wording. Acceptance: cases 02--10 retain the documented +1 legacy
  offset; case01 covers the clipped-z 109/110-slice difference; synthetic crop
  and coordinate-round-trip checks pass; images/masks and all 300 landmarks
  remain aligned; legacy outputs match their frozen source geometry. Recording
  this task did not launch optimization; read-only real-data parity and
  isolated synthetic preparation checks passed.
- [x] R05: Freeze artifact inputs/checksums, aggregation/SD conventions,
  captions, method mapping, baseline source roles, and public column schemas.

## P1 — Build a portable release

- [x] R06: Extract local methods and three upstream wrappers; retain behavior
  through focused synthetic numerical checks and exact upstream pins. Remove
  dependence on private `AGENTS.md`, absolute paths, and experiment directories.
- [x] R07: Package raw DIR-LAB loaders, R231 mask preparation, 1 mm preprocessing,
  coordinate transforms, and validation. Check against existing prepared data
  without rerunning registration. R231 interface/API is checked; inference was
  not re-executed and no weights are bundled.
- [x] R08: Extract CPU-only Figure 2/table renderers with bundled frozen values.
  Include external reference dots without their training/inference environments.
  Display static Figure 1; defer Figure 3 and diagnostic reproduction code.
- [x] R09: Audit actual dependency closure; choose a compatible pinned runtime
  or explicitly isolated targets. Add Dockerfile, dependency lock, and
  `.dockerignore`; verify a fresh build and import/GPU visibility.
- [x] R10: Write tested running instructions: clone, build, data, preprocessing,
  per-pair method/evaluation commands, frozen figure/table commands, outputs,
  resources, and troubleshooting. Mark unexecuted commands honestly.
- [x] R11: Update public README with Figures 1/2, local poster link, actual
  release status, citation, and the paper/update distinction. Preserve existing
  Figure 3 asset unless separately asked to remove it.
- [x] R12: Add citation/project license/third-party notes and results provenance;
  audit the exact public file list and Git history for unintended private data.

## P2 — Acceptance and handoff

- [x] R13: From an allowlisted clean source snapshot, regenerate tables and Figure 2
  without raw scans, private mounts, internet at plotting time, or optimization.
  Compare numeric rows/rounding and approved visual layout with frozen targets.
- [x] R14: Verify preprocessing geometry/landmark coverage, wrapper numerical
  contracts, package imports, GPU visibility, and documentation links. State
  explicitly that full paper optimization has not been rerun.
- [x] R15: Review exact deliverables and remaining limitations. Submodule add
  automatically stages the three gitlinks and `.gitmodules`; other files remain
  unstaged/untracked. No commit, tag, image publication, or GitHub push is made.

Do not run the full sweep to make this checklist green. Existing user changes
in experiment code and governance docs are outside this audit's edit scope.

Implementation acceptance is complete; this is not a publication confirmation.
Evidence and deliberately unexecuted paths are recorded in PROGRESS.md and
progress.json. Maintainer review, commit/tag/push and image publication remain
separate actions requiring direction.

## Follow-up — both reproduction layers

- [x] R16: Audit exact three-dataset experiment matrices, baseline scope, seeds,
  schedules, cohort IDs and per-source SD conventions.
- [x] R17: Implement cohort runner, verified resume, preserved failed attempts,
  independent-job sharding, persistent paired timing and strict new-result
  aggregation/plots. Keep frozen values immutable.
- [x] R18: Complete clean Docker acceptance of new code and synchronize reports;
  real registration optimization remains prohibited for this task.

## Publication hardening — 2026-09-30

- [x] R19: Synchronize pending SINR permission and the explicit publication hold.
- [x] R20: Prepare the verified OASIS ordinal-ID map, safe organizer and instructions.
- [x] R21: Refresh checksums and test a fresh recursive clone of a disposable candidate commit.
- [x] R22: Stage only the explicit complete public payload; preserve official branch/history.
- [ ] PERMISSION-001: Obtain SINR clearance or approve a separately reviewed alternative scope.
- [ ] Publish only after permission resolution and explicit commit/push authorization.
