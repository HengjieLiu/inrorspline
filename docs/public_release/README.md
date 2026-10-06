# RightPriorDIR public release plan

Status (2026-09-30): both reproduction layers implemented; publication on hold
because SINR author permission is pending. Technical hardening, fresh candidate
clone/Docker acceptance and complete scoped staging are finished;
see [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) and
[publication_readiness.json](publication_readiness.json). This folder is internal release planning in
the private research repository. The public destination is the existing sibling
`RightPriorDIR` checkout. Do not export this folder or private Git history.

The user subsequently authorized implementation and selected MIT for own code.
The package, frozen rendering, versioned preparation and Docker environment are
implemented in the public checkout. No registration optimization, commit or
publication is performed. The initial audit below is historical; current state:

- [PROGRESS.md](PROGRESS.md): Chinese human-readable milestone report.
- [progress.json](progress.json): machine-readable milestone evidence and limits.
- [EXPERIMENT_REPRODUCTION.md](EXPERIMENT_REPRODUCTION.md): follow-up two-layer audit, implementation and acceptance.
- [experiment_reproduction.json](experiment_reproduction.json): machine-readable full-cohort rerun scope and evidence.

- [AUDIT.md](AUDIT.md): evidence, gaps, and source locations.
- [TODO.md](TODO.md): implementation order and acceptance criteria.
- [release_spec.json](release_spec.json): machine-readable scope and decisions.

## Confirmed scope

| Item | Release decision |
| --- | --- |
| Third-party methods | IDIR, SINR, and Dual-INR as pinned upstream submodules with local wrappers. |
| SINR permission | Pending, explicitly confirmed by the user on 2026-09-30. Supersedes the earlier confirmed-permission statement. Public repository/image release is on hold pending clearance or an approved alternative scope. |
| Existing optimization results | Reuse frozen 2,500-step results; do not rerun optimization to prepare this release. |
| Reproducibility | Include frozen numerical inputs and figure/table generation scripts. |
| Diagnostics | Defer target-field fitting/Figure 3 and diagnostic experiment reproduction code. |
| Public main table | Replace the displayed table with the selected updated acceleration results; explain differences from the published paper. Preserve the original as a separate historical record. |
| OASIS update, final explicit choice | September 16 five-method base/fast **paired timing**, labeled **N=10**; historical **OASIS100 accuracy is listed separately**. September 24 CP-BE results are not selected. |
| Lung acceleration | Pending publication selection; do not silently promote newer completed experiments. |
| COPD | Include preprocessing, configurations, and existing verified results as a post-paper extension; no requirement to fill gaps by rerunning. |
| External reference dots | Freeze values and provenance for VoxelMorph, TransMorph, VFA, SITReg, Greedy, and pTVReg; their executable environments are outside the first release. |
| README | Display Figures 1 and 2, link the paper and poster, show precise release status and an updated-results note. |
| Environment | Provide a buildable Dockerfile, locked dependencies, and detailed commands covering our methods, the three INR baselines, preprocessing, metrics, and plotting. |
| Own-code license | MIT; upstream licenses remain separate and SINR permission is still pending. |

The original Figure 2 remains frozen. Own-method base configs use 2,500 steps;
implementation found that lung Dual PL1A uses the author's 3,000-step blur
schedule, now explicitly disclosed. Replacing
the public acceleration table does not replace Figure 2 or rewrite the accepted
paper. Use distinct result-set IDs and captions for these artifacts.

## Public payload

Implemented paths in the public checkout:

```text
RightPriorDIR/
  README.md
  LICENSE                         # MIT for own code
  CITATION.cff
  THIRD_PARTY.md
  Dockerfile
  .dockerignore
  .gitignore
  requirements.txt                # pinned overlay on a digest-pinned base
  release_manifest.json           # explicit file allowlist + hashes
  rightpriordir/
    __main__.py                   # frozen, prepare-oasis/lung, masks, register/evaluate, experiment/summarize, verify
    oasis*.py                     # OASIS loader, math, registration engine
    lung*.py                      # lung loader, math, registration engine
    geometry.py, prepare_lung.py   # explicit legacy/corrected crop contracts
    upstream.py, dual_*.py         # checked upstream imports and local adapters
    frozen.py, _plot_*.py          # CPU-only frozen artifact reproduction
  third_party/{IDIR,SINR,DUAL-INR-DIR}/
  third_party/lock.json
  configs/{oasis,4dct,copd}/        # 15 runnable base/extension configurations
  experiments/*.json              # 5 complete quality/timing suite definitions
  splits/                         # explicit fixed/moving pair manifests
  results/{paper,updated_oasis,copd_extension}/
  figures/{fig1.png,fig2.png,fig3.png}
  poster/offgrid_poster_v3.pdf
  docs/{running,data,methods,results,validation}.md
  docs/{environment,source_extraction}.json
  tests/test_release.py
```

Preserve existing public-repository history and assets. Figure 3 is already
tracked there: excluding it from README and the release checklist does not
authorize deleting the file or rewriting history.

Selected small frozen CSV/JSON and figure/poster assets are intentional public
inputs. Their export does not change the private repository's external-artifact
policy. Never copy entire experiment folders, logs, worker launchers, private
instructions, host paths, raw scans, masks, fields, checkpoints, or credentials.

## What reproducibility means in this release

1. **Paper artifacts:** render Figure 2 and paper/updated/extension tables from
   bundled frozen numerical inputs, without raw data, a GPU, or optimization.
   Figure 1 is the paper overview illustration; retain its static asset and
   identify any available editable source separately.
2. **Runnable methods:** expose simple documented registration/evaluation
   commands for the included methods and fixed configurations. Distinguish
   user-invoked optimization from the frozen-result plotting path. Publishing
   these commands does not require rerunning the paper sweep during release
   preparation.
3. **Deferred work:** Figure 3, target-field fitting diagnostics, exploratory
   sweeps and cache/backend diagnostics are outside
   the payload. Necessary provenance for an included result is still included.

The user's follow-up makes layer 2 a first-class deliverable: complete selected
experiment matrices, resumable cohort execution, paired OASIS timing, strict
aggregation and new-result plots are now implemented. This does not authorize
executing them during preparation. The new guide is public `docs/experiments.md`.

No claim of independently rerun paper experiments, bitwise GPU determinism, or
identical runtime on other hardware. Numerical agreement and plot formatting
need explicit, appropriate tolerances; frozen source hashes must match exactly.

## Data contracts

### OASIS

- Paper figure cohort: seed-3 shuffled `pair_test_200`, ranks 0--99, **100
  ordered pairs**. Source: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Pair order is **fixed ID, moving ID**; preserve zero-padded IDs, rank, and
  original index. Export the actual rows plus a checksum, not only a random seed.
- September 16 timing update: ranks 0--9, separately identified as N=10.
  Its first10 Dice is diagnostic context, not the historical OASIS100 accuracy
  block. The latter retains its own source and backend provenance.
- Specify the exact prepared-data source/version and directory mapping:
  `test/img<ID>.nii.gz`, `test/seg<ID>.nii.gz`, and required split metadata.
  Do not imply that any raw OASIS download is directly interchangeable.
- Release IDs are one-based ordinal positions in Neurite `subjects.txt`, not
  source subject-number digits: 0333 maps to OASIS_OAS1_0370_MR1, 0115 to
  OASIS_OAS1_0123_MR1. All 84 test-pool subjects / 168 original compressed
  volumes matched historical inputs exactly. Public `prepare-oasis` pins this
  mapping/checksums and copies without voxel transforms.
- Freeze orientation handling, expected post-loader shape `(160,224,192)`,
  pair normalization, the 35-label metric set, missing-label rules, and
  raw-image-positive foreground (`fixed_raw > 0`).

### DIR-LAB 4DCT and COPD

- Both use cases 01--10. 4DCT: moving T50, fixed T00. COPD: moving eBHCT,
  fixed iBHCT. Describe stored pull-field direction separately.
- Publish raw-file layout, per-case shape/spacing, raw dtype/byte order,
  intensity encoding and normalization, XYZ/ZYX order, one-based landmark
  conversion, crop bounds, resampling shape rounding, interpolation,
  `align_corners`, transformed landmarks, and mask provenance.
- Existing preprocessing uses the union of fixed/moving landmarks and margins
  XYZ `[10,10,5]`, with PyTorch trilinear image interpolation and nominal 1 mm
  grids. Explicitly disclose landmark-dependent cropping and the distinction
  between nominal spacing and the implemented endpoint mapping.
- **Keep crop variants distinct.** Current 4DCT preprocessing uses corrected
  MATLAB-origin bounds; current COPD preprocessing intentionally uses legacy
  Python bounds. Preserve the convention used by each frozen result. Do not
  silently correct historical inputs while retaining their old output numbers.
- **Implemented fix: CROP-001 / R04.** Public preprocessing now requires an
  explicit crop mode and writes distinct protocol directories. Corrected COPD
  is available as a separate preparation protocol, not as new frozen results.
  The private TRE standard is corrected; historical inputs/results remain
  untouched. All 20 cases passed geometry/landmark parity checks, with image and
  mask parity checked on both phases of case01/02 in each cohort.
- Both preprocessors currently require lung masks. Include reproducible R231
  mask preparation and weight/version handling, or explicitly separate an
  image-only preparation path from mask-dependent method/metric paths.
- Headline TRE: native/full-or-crop physical mm with snap-to-voxel convention;
  retain native subvoxel and registration-grid metrics with explicit labels.
  Record foreground domain, folding rule, Jacobian handling, aggregation unit,
  SD convention, and all 300 landmarks per case.
- Public preprocessing must be tested against existing prepared artifacts,
  including coordinate round trips. This does not require registration reruns.

## Wrappers and provenance

The three inspected commits are now release gitlinks with tracked-file hashes.
This verifies the shipped source, not every historical experiment's upstream
revision: the old IDIR/SINR paths contained local SIREN implementations.
The migrated SIREN and spline kernels passed exact numerical comparisons.
Document wrapper-level changes: preprocessing, sampling, loss/regularization,
optimizer, stage budgets, coordinate units, and dense-field evaluation.

The public wrappers use upstream SIREN networks or the Dual-INR model while
retaining local loss/spline/training semantics where needed. Adaptations are
documented in public `docs/methods.md`; they are not described as entirely
unmodified upstream pipelines. BE normalization and alpha scales are preserved,
generic-module imports are isolated, and upstream worktrees remain clean.

For each frozen artifact record: result-set ID, paper role, source files and
SHA-256, cohort, method/configuration ID, implementation/source hash, upstream
pin, preprocessing ID, metric convention, backend flags, and selection rule.
No per-case best-method or best-hyperparameter cherry-picking. Preserve
published aggregation/rounding rules rather than recomputing them differently.

## Docker and running instructions

The public Dockerfile uses the audited historical public base pinned by digest
and a version-pinned package overlay. All three adapters share this environment;
import/numerical checks and `pip check` pass. Upstream requirement declarations
are recorded as historical conflicts rather than installed wholesale. Anonymous
access to the base digest was verified. See public `docs/environment.json`.

The implemented public `docs/running.md` provides commands for:

1. Recursive clone/submodule initialization and pin verification.
2. Docker build, CPU plotting check, GPU visibility/import check.
3. Dataset acquisition and layout validation, mask setup, 4DCT/COPD preprocessing.
4. One-pair registration with a selected fixed configuration, then evaluation.
5. Regenerating paper Figure 2, the selected updated OASIS table, and verified
   COPD tables from frozen data without entering any optimizer.
6. Output paths, expected filenames, elapsed-time interpretation, tested memory
   usage, and common missing-data/submodule/weight errors.

Use portable raw-data read-only, prepared-data, and output mounts; run as the
host user's UID/GID. Document host Docker/NVIDIA driver/Container Toolkit
requirements separately from image contents. Record required model downloads
and checksums; do not hide first-run network access. No private mounts or
credentials may be needed. Do not advertise measured VRAM as a verified minimum.

The original audit ran no GPU workload. Implementation acceptance includes a
scalar CUDA visibility check, imports, frozen-result plotting, synthetic
numerical contracts and preprocessing parity; optimization and R231 inference
are explicitly not rerun. The guide distinguishes tested and unexecuted paths.

## Maintenance

`release_spec.json` owns stable decision IDs, selected versions, pins, artifact
states, and task states. This README explains the policy; AUDIT.md records
evidence; TODO.md is the human checklist. Update affected files together and
never turn a recommendation or unanswered question into a confirmed decision.

The old `public/release_manifest.md` describes an anonymous research scaffold
and is not an export manifest for this named, accepted-paper release.
