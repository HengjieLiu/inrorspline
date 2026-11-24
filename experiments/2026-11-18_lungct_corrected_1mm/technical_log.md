# Technical log

## Known limitations

Historical legacy reuse limits causal attribution to cropping. CorrField uses
raw HU plus fixed R231 masks; pTV/MR use clipped normalized intensities without
lung-mask weighting. Nominal1mm endpoint mapping and Python shape rounding
are preserved, including corrected COPD01=272 slices.

## Execution

- 2026-11-18: approved implementation and30 production registrations.
- Preserved initial unrelated dirty worktree and existing containers/results.
- New isolated pinned-image runtime created; dependency overlay is external.
- External execution key: `lungct_corrected_1mm_20261119T073622Z`.
- Preflight, launch, acceptance and results will be recorded below.

- Fresh runtime passed four-GPU CUDA allocation/import checks. Image ID and
  mounts are captured externally; isolated overlay pins SimpleITK2.4.1,
  h5py3.12.1 and pytest8.3.5 without modifying the image/default environment.
- All20 prepared products passed finite-array, geometry and round-trip checks;
  every historical4DCT array matched exactly. Boundary/interior HU and mask
  reconstructions passed. Retained product size approximately9.9GiB.
- All40 historical method/case records reproduced their saved native TRE;
  historical native field copies and receipts occupy approximately1.8GiB.
- Twenty-one geometry, matrix and existing-regression tests passed in Docker.
- Campaign controller launched after preparation/reference acceptance; source,
  input, runtime and parameter receipts precede the GPU preflight queue.
- All four preflight registrations passed. Both largest-case CorrField fields
  were reused as production results; peak allocated memory5.51/6.21GiB.
- Thirty production attempts yielded27 accepted fields: COPD MR10,
  CorrField4DCT7/COPD10. Original CorrField4DCT04/05/06 failed with IndexError
  in minimum_spanning_tree; failed directories/logs are retained. The controller
  completed other jobs and exited with failure after collecting all three.
- Completed COPD MR native snap mean: legacy1.0501+/-0.7331mm,
  corrected1.0436+/-0.7462mm; paired difference -0.0065+/-0.0542mm.
  COPD CorrField rawHU/fixedR231 mean2.1808+/-1.3359mm. All SDs here are
  sample SDs across ten case means; within-case report SDs were independently
  recomputed with ddof1 rather than copied from historical population-SD fields.
- Independent posthoc audit accepted27 cases/8100 paired landmarks with exact
  native snap/subvoxel TRE agreement and preserved raw/prepared/source hashes.
  Ordered reports contain67 evaluated method/case records and20100 errors;
  4DCT CorrField04/05/06 remain explicitly failed, with no complete-cohort mean.
- A separate optional `forest_guard.py` supports independent message passing
  on existing graph components, adding no edges or parameters. Connected paths
  are bitwise upstream-equivalent in tests. Three guard tests passed. User choice
  was requested before enabling it for the failed cases; no guarded registration
  has been run and no original numerical source was modified.
- 2026-11-19: user approved the component-wise guard for only4DCT04/05/06.
  Recovery uses fresh `forest_guard_v1` case directories, a frozen manifest,
  per-call connectivity traces and a report-input assembly of accepted cases.
  No original upstream/runner/evaluator files are modified by this recovery.
- Guarded recovery accepted4DCT04/05/06 with original snap mean/sample SD:
  1.3768259306/1.3072343179,1.1545013252/1.3707137596,
  1.0456328515/0.9462237603mm. Component-wise handling occurred in3/4,2/4,
  and4/4 graph calls respectively; no new edges or parameter changes.
- Full4DCT CorrField cohort1.0423+/-0.1577mm; COPD cohort unchanged at
  2.1808+/-1.3359mm. CSV/JSON tags and report disclosures identify the three
  guarded rows. Earlier reports: forest_guard_v1/previous_reports.
- Final independent audit accepted30 cases/9000 paired landmarks with exact
  snap/subvoxel errors and SciPy resize error below2e-4 native voxels. Separate
  recovery audit verified frozen guard/wrapper hashes, all12 graph traces,
  67 unchanged prior evaluations, original failed attempts and archived report
  checksums. All70 evaluations and21000 landmark metric rows are retained
  externally. Thirty unique full endpoints accepted after33 full attempts plus
  2 reduced MR smokes; no commits/pushes.
