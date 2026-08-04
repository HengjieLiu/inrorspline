---
doc_type: technical_log
status: complete
experiment_id: dual_inr_dirlab_controlled_multires_v1
date_planned: 2026-08-02
date_completed: 2026-08-03
---

# Technical Log

## Known limitations and warnings

- N and P are different preprocessing protocols. Only N0-N1 and P0-P1 isolate
  the image-pyramid schedule; N-P is contextual and not method-only evidence.
- N uses raw native anisotropic full volumes and a fixed/T00 R231 mask to choose
  optimization coordinates. P uses a landmark crop resampled to 1 mm and no
  lung mask in optimization. These domain differences remain even though model,
  objective, seed, stage timing, and optimizer policy are shared.
- N1/P1 reproduce the shape `[8,4,2,1]` of a coarse-to-fine schedule, not the
  MR-D-BSCP representation. There is no B-spline control lattice, `32-16-8-4`
  control spacing, knot insertion, isoTV objective, or transfer of its tuned
  regularization weights.
- The same `0.01/0.01` normalized-coordinate regularizers do not have identical
  physical scale on anisotropic native N and isotropic P grids. This is a fixed-
  hyperparameter transfer baseline, not evidence that either protocol is tuned.
- The Gaussian sigma is defined in source-grid voxels, so N's anisotropic
  pyramid is not an isotropic physical-space scale-space. This caveat must
  accompany any N0-N1 interpretation.
- Full-resolution optimization domains are sampled at every level. The R231
  mask and P five-voxel border are not downsampled; only normalized queries are
  evaluated on the stage image.
- P's fixed-lung foreground is allowed only for posthoc regularity measurement.
  Using it for sampling, similarity, loss weighting, or early stopping would
  violate the locked P protocol.
- Voxel-coordinate diffusion and bending are not physically interchangeable
  across N and P. All regularity rows require protocol and metric-domain tags.
- Seed 42 over ten cases is a manageable camera-ready baseline, not a multiseed
  robustness analysis.
- No TRE or regularity value is an acceptance gate. A poor result remains a
  valid completed result and must not cause a job or the nightly queue to stop.

## Locked decisions

- Cases `01-10`, direction T50 to T00, seed `42`; 40 total records.
- Source commit:
  `f7c8e833435ec9c6b7ebec6c9a8098b566709146`.
- N0/N1: native full volume, native intensities, no crop/resample, fixed-T00
  R231 coordinate domain; factors `[1,1,1,1]` versus `[8,4,2,1]`.
- P0/P1: canonical pTVReg 1 mm landmark crop, `[80,900] -> [0,1]`, no lung
  optimization mask, five-voxel border; the same two factor schedules.
- All: DUAL SIREN `3-256-256-3`, omega `10/30`, output scales `0.1/0.05`,
  30,000 samples, NCC + `0.01` Laplacian + `0.01` negative-Jacobian, fine
  regularization scale `0.1`, semantic weight zero.
- The fine-scale constructor value is `0.05`; at step 400 the released
  `turn_on_fine_branch()` rule resets it to `0.1` times the then-current coarse
  scale. The stage-local warmup scheduler initializes the effective LR at zero,
  so the first optimizer update in each stage is a zero-LR warmup update.
- Stage lengths `[400,400,800,900]`, starts `[0,400,800,1600]`, fine branch
  activates at step 400. Model persists; AdamW and warmup-cosine scheduler are
  fresh per stage (`1e-3 -> 1e-5`, weight decay `0.01`, warmup 20).
- Four GPUs, one optimizer per GPU, dynamic queue. All 40 keys are attempted;
  technical failures are retried without pausing unrelated jobs.

## Initial audit evidence

- The completed author-reproduction experiment already exercised the pinned
  DUAL source with the declared Docker environment across all ten cases. Its A0
  author-protocol result was `0.97894 +/- 0.17273 mm` snap TRE (sample SD), and
  A1 completed with `0.98343 +/- 0.16688 mm`. These values are provenance and
  implementation references only; the author mask-derived crop is not reused
  as N or P preprocessing.
- Code inspection found that the released author trainer changes Gaussian blur
  but does not construct the true downsampled image levels required here. The
  controlled implementation therefore needs an adapter with explicit,
  independently generated stage tensors while leaving the external repository
  unchanged.
- Existing project protocol assets distinguish raw native fixed-mask data from
  the canonical pTVReg 1 mm crop and provide native-remapping evaluation. The
  launch still requires a fresh ten-case manifest, geometry/landmark checks,
  mask-domain checks, and round-trip coordinate audit before optimization.
- The current execution target is the project Docker container and four-GPU
  dynamic queue. Live container/GPU capacity and writable output-root checks
  remain mandatory immediately before launch.

## Run ledger

| Stage | Status | Required evidence |
|---|---|---|
| pinned-source and environment audit | passed | exact clean commit, running declared container, four idle RTX 6000 Ada GPUs |
| native and pTVReg data audit | passed | 10/10 cases per protocol, geometry, 300 landmarks, fixed-mask/border domains validated |
| adapter and focused tests | passed | controlled suite 10/10 in Docker; compile and all-case loader checks passed |
| 40-key manifest dry run | passed | 40 unique full keys, four smoke keys, four GPUs, external output root |
| case01 four-variant smoke | passed | 4/4 complete; exact levels/fingerprints, fine activation, dense/native metrics finite |
| cases01/08 technical runs | folded into cohort | user authorized overnight all-40; case-major queue starts four case01 arms and does not performance-gate later cases |
| four-GPU full cohort | passed | 40/40 unique records completed on attempt 1; zero failures, retries, duplicates, or missing keys |
| aggregation and visual audit | passed | strict 40/40 aggregation; TRE, regularity, runtime, reference, and paired schedule tables complete |

## Execution notes

Append dated launch commands, source/data/output hashes, container and GPU
identity, retry decisions, accepted record keys, and aggregation checks here.
Never revise the locked scientific configuration in place. Preserve all
nonpromising scientific outcomes and all technical-failure evidence.

### 2026-08-03 launch notes

- First smoke attempt exposed a CUDA peak-memory API argument incompatibility
  before training. No scientific output was accepted; all four incomplete
  directories and launcher logs were preserved. The runner was repaired to set
  the CUDA device explicitly and use the integer device index for memory
  accounting.
- The repeated four-arm case01 smoke completed `4/4` in approximately 70-75 s
  per record. N0/N1 and P0/P1 had identical stagewise coordinate fingerprints;
  N1/P1 resolved the expected ceil-shaped `[8,4,2,1]` levels. Smoke used one
  warmup step per stage, so its TRE values are technical diagnostics only.
- The full queue started under
  `${RUN_ROOT}` on GPUs
  `0,1,2,3`, with at most one optimizer per GPU and two technical attempts per
  key. Core optimization sources were frozen at launch. Aggregation/reference
  code is intentionally outside the per-case source bundle and may be repaired
  posthoc without changing any optimization record.
- The planned standalone full-length cases01/08 pilot gate was folded into the
  all-40 case-major queue because the user explicitly requested every arm run
  overnight regardless of scientific promise. This changes execution order,
  not any data, model, loss, schedule, seed, or metric setting.
- The full queue finished at `2026-08-03 07:04:16 UTC`, approximately 57 min
  after launch. Every one of the 40 keys completed on attempt 1 with return
  code zero; all four variants contain ten accepted cases.
- The strict aggregate finished at `2026-08-03 07:13:01 UTC` without
  `--allow-partial`. Common posthoc regularity is complete for 40/40 records,
  and both external reference cohorts contain all ten cases. MR-D-BSCP uses a
  single cohort-selected `w=0.055` setting for every case; per-case oracle
  selection is explicitly false.
- Final headline/native snap TRE (mean +/- sample SD, mm) is N0
  `1.0738 +/- 0.2041`, N1 `1.0649 +/- 0.2091`, P0
  `1.0410 +/- 0.2160`, and P1 `1.0461 +/- 0.2211`. Paired schedule
  deltas are N1-N0 `-0.0089 +/- 0.0134 mm` and P1-P0
  `+0.0051 +/- 0.0147 mm`. The pyramid schedule therefore has a small,
  direction-inconsistent effect under this frozen recipe.
- Common foreground regularity (diffusion / bending / SDlogJ / J<=0 percent)
  is N0 `0.0134 / 0.0048 / 0.1282 / 0`, N1
  `0.0135 / 0.0046 / 0.1286 / 0`, P0
  `0.0059 / 0.0002 / 0.1135 / 0`, and P1
  `0.0060 / 0.0002 / 0.1138 / 0`. Voxel-coordinate diffusion and bending
  remain non-comparable across native N and isotropic P preprocessing.
- Contextual 1 mm references are pTVReg `0.9191 +/- 0.1607 mm` and fixed
  MR-D-BSCP `w=0.055` `0.9175 +/- 0.1561 mm`. Their separate
  preprocessing and optimization tags are retained in the camera-ready
  readout; they are not silently treated as method-only comparisons.
- Final integrity audit found one frozen source-bundle SHA256 across all 40
  runs (`4abe7aa9fa09632a11dbace5e3277fbe08dd84fed8461c64f61fb11ab74c934e`),
  2,500 contiguous finite history rows per run, exact locked stages/factors,
  matched within-case coordinate hashes, finite dense fields, and close
  dense/direct subvoxel TRE agreement. No axis, field, or schedule defect was
  detected.
- Before commit, machine-local paths in the compact review bundle were
  normalized to symbolic roots (`${RUN_ROOT}`, `${REVIEW_ROOT}`, and input-root
  placeholders). This serialization-only change did not alter numeric results.
