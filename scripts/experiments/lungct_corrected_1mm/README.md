# Corrected 1 mm LungCT campaign

This adapter preserves the existing MR-D-BSCP numerical runner and the pinned
standalone CorrField implementation. It does not modify either method's loss
or optimization defaults beyond the explicit campaign arguments.

Use the standard pinned-image Docker runtime with a read-only repository,
read-only DIR-Lab data, read-only CorrField source at `/opt/corrfield-source`,
and writable `ARTIFACT_ROOT`. The source root is also configurable by CLI;
no private host path is part of the interface. Runtime dependency versions
and the separate dependency overlay are recorded externally.

The campaign has three stages:

1. `prepare.py`: creates fresh versioned corrected 4DCT/COPD inputs with pTV
   normalized and unclipped HU profiles; checks all20 geometries and exact
   historical4DCT parity. Crop modes/defaults remain explicit in the two
   preprocessing CLIs. Existing case directories are never overwritten.
2. `references.py`: replays the audited pTV and MR fields in native physical
   coordinates, verifies saved TRE, and retains per-landmark records externally.
   Pass the earlier availability audit as `--audit`; do not infer casewise
   best settings or substitute the frozen COPD w0.11 cohort.
3. `campaign.py`: requires accepted preparation/reference receipts and
   `runtime.json`, freezes source/input hashes, runs reduced COPD MR smokes
   and full largest-case CorrField smokes, then completes the30-case queue
   with one process/GPU. Accepted CorrField smokes are reused. Partial/failed
   attempts are retained; unaccepted existing output paths block overwrites.

All CLIs expose their required roots with `--help`. `campaign.py` requires
`--data-root`, `--run-root`, `--review-root`, `--reference-root`, and
`--preparation-receipt`; `--gpus` defaults to0,1,2,3.

## Field and metric contract

`disp_vox_reg1mm_zyx.npy` is FP32 `[Z,Y,X,3]` with ZYX displacement components
in nominal1mm registration voxels and `moving(fixed+u)` pull semantics.
CorrField receives XYZ tensors, so its XYZ/channel-last result is transposed
and its components reordered explicitly before export. MR canonical files
link to their retained solver output within the new run directory.

`disp_vox_native_crop_zyx.npy` uses native voxel components on the native crop.
`evaluation.json` records origin, spacing, crop/registration shapes, source
hashes, all four TRE modes and the method's backend/input/mask profile.
`per_landmark.csv` retains absolute native points, displacements, predicted
points and errors. Original/native snap TRE is headline; registration-grid
TRE is diagnostic. Crop/full native TRE must agree exactly.

## Reporting and limitations

The report order is COPD legacy/corrected/delta, 4DCT three-method, then COPD
three-method. Each case uses mean/sample SD over300 landmarks; aggregates
use mean/sample SD over10 case means. `reports/accepted.json` seals70 evaluated
method/case records (30 new plus40 references) and21,000 landmark metric rows.

MR retains the historical cuDNN T/F override. CorrField uses default F/T,
raw-minus1024 HU and fixed R231 masks; MR/pTV use clipped intensities without
lung-mask optimization. Historical reuse is not a contemporaneous crop-only
ablation. Python np.rint shape rounding is retained (corrected COPD01=272
slices). No result-driven retries, forced similarity, or parameter tuning.

## Approved disconnected-graph recovery

The original CorrField attempts for 4DCT04–06 failed on disconnected graphs.
After explicit user approval, `recover_forest.py --review-root <review-root>
--gpus 0,1,2` recovered only those three cases in fresh `forest_guard_v1`
directories. It verifies the original source hashes, freezes the recovery
sources, and retains the failed attempts and prior reports. Existing recovery
directories block a second launch.

The guard runs upstream MST/message passing independently within each existing
graph component. Connected graphs follow the original path; no edges or
parameters change. Guarded evaluations carry the implementation tag
`corrfield_componentwise_forest_v1` and component traces. CSV/JSON tables and
the report disclose the three guarded rows.

After recovery, `campaign_accepted.json` is the final campaign receipt and
`forest_guard_v1/assembled_cases` selects all 30 accepted outputs. The original
campaign failure state remains a historical record. Run the independent audit
with `scripts/reports/audit_lungct_corrected_1mm.py --review-root <review-root>
--run-root <review-root>/forest_guard_v1/assembled_cases` to check the complete
cohort, including the recovery fields.
