# Corrected 1 mm LungCT registration campaign

Approved 2026-11-18. Execution ID `lungct_corrected_1mm_20261119T073622Z`.
Thirty production registrations: corrected COPD MR-D-BSCP10 and raw-HU
CorrField4DCT10/COPD10. Reuse historical legacy COPD w0.055, corrected 4DCT
w0.055 and both pTVReg cohorts; no fresh legacy cohort or hyperparameter search.

## Known limitations

Historical comparison is not a contemporaneous crop-only ablation. CorrField
uses unclipped HU and fixed R231 masks, unlike pTV/MR normalization and no lung
mask in optimization. Corrected COPD01 is272 slices under existing NumPy rint
rounding, versus275 legacy and273 in the MATLAB pTV archive. Preserve old data.

## Frozen settings

- Corrected MATLAB-origin union-landmark crop, margin XYZ10/10/5; trilinear
  align_corners=True, np.rint(shape*spacing); masks nearest. HU=raw-1024.
- COPD MR: controls64/32/16/8/4, downsample16/8/4/2/1,
  epochs400/400/400/400/900, Adam1e-3, seed1, FP32, exact knot insertion;
  signed pTV-LCC sigma2.1mm plus physical control isoTV0.055, csqrt0.005,
  full-resolution border5. Explicit historical cuDNN override benchmark=True,
  deterministic=False, passed as --cudnn-benchmark --no-cudnn-deterministic.
- CorrField: standalone archive SHA256
  a2aaa91add1286ef3cc77a0bf049a4d3ceed57dd4dfa4bcd39b7cac559a57472,
  existing torch.linalg.solve compatibility fix, alpha2.5 beta150 gamma5 delta1
  TPSlambda0 sigma1.4 sigma1=1, L16/8 N6/3 Q2/1 R3/2 Tn/n. Fixed R231 only,
  no moving-mask/landmark inputs. Seed1 FP32, cuDNN benchmark=False,
  deterministic=True. No tuning or parameter-changing retries.
- New runtime: same pinned image digest as private/data_sources.md, configured
  host UID/GID, RO repo/raw/shared/CorrField mounts, RW ARTIFACT_ROOT.
  Dependency overlay is separate from the image and recorded in provenance.
- One worker per idle GPU; no stopping or repurposing existing containers/jobs.

## Outputs and acceptance

Prepared root ${ARTIFACT_ROOT}/prepared/<execution-id>; run root
${ARTIFACT_ROOT}/runs/<execution-id>; review root
${ARTIFACT_ROOT}/repo_experiments/<execution-id>. All generated material stays
external. Source/config hashes, image/container/software/input identities and
requested/effective backend settings are recorded before production.

Preflight: geometry/field unit tests; all20 real coordinate round trips;
4DCT prepared-array exact parity; historical TRE replay; reduced COPD01/02
MR smokes; full largest-volume 4DCT02/COPD10 CorrField smokes counting as
production outputs. Require finite expected-grid fields, reload agreement,
300 landmarks, native/crop evaluation agreement and available disk/memory.
TRE differences, including worse results, do not trigger tuning or rejection.

Save FP32 [Z,Y,X,3] ZYX pull displacements, native-crop voxel fields,
per-landmark coordinates/displacements/predictions/snap/subvoxel errors, all
four TRE modes, source/geometry receipts and process timing/memory.

Report order: COPD legacy/corrected/delta; 4DCT pTV/MR-corrected/CorrField;
COPD pTV/MR-corrected/CorrField. Headline native snap mm; within-case sample SD
over300 landmarks, aggregate unweighted mean/sample SD over10 case means.
Include diagnostic metrics and protocol caveats. No commit/push/publication.

## 2026-11-19 approved recovery addendum

The user explicitly approved the tested component-wise graph guard for the
three failed original CorrField4DCT04/05/06 attempts. Rerun only these cases
under `forest_guard_v1/`, retaining the failed original directories and a
snapshot of partial reports. All images, masks, keypoints, scalar/stage
parameters, seed, backend and source modules remain unchanged.

`forest_guard.py` runs upstream MST/TBP independently on existing connected
components, preserving every graph edge/cost and isolated unary term. The
connected path is bitwise upstream-equivalent in tests; no bridging edges
or parameter adjustments are introduced. Tag the recovered fields
`corrfield_componentwise_forest_v1` with all four stage/direction graph traces.
Freeze the guard, wrapper and recovery/audit source hashes before launch.

Assemble30 accepted case directories without replacing originals; regenerate
the three ordered tables with explicit guarded-case disclosure. Independently
audit all30 fields/9000 paired landmarks and preserve the previous27-case audit.
The campaign has33 full attempts for30 unique accepted case endpoints after
this approved recovery, plus the two reduced MR smokes.
