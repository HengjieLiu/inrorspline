# Corrected 1 mm LungCT MR-D-BSCP and CorrField

Source: `scripts/experiments/lungct_corrected_1mm/`; protocol and authorized
scope: `experiments/2026-11-18_lungct_corrected_1mm/codex_execution_spec.md`.

## Preparation and historical compatibility

Both dataset preparers expose `--crop-convention corrected|legacy` and
`--save-hu`. Defaults remain corrected4DCT and legacyCOPD. Existing case
directories are rejected even if an old caller passes `--overwrite`; a new
versioned destination is required. Frozen datasets are never rewritten.

Corrected bounds compute the fixed/moving landmark union in one-based XYZ,
add margins10/10/5, clip to image bounds, then convert both inclusive endpoints
to zero-based ZYX. Legacy bounds retain the old missing-minus-one behavior.
Target shapes retain NumPy rint(crop_shape*native_spacing), images use
trilinear align_corners=True and masks use nearest. Corrected COPD01 has109
native z slices and272 registration slices, versus110/275 in the old Python
product and273 registration slices in the MATLAB reference.

HU profiles are raw-minus1024 before crop resampling, without clipping.
MR intensity profiles retain clipping80..900 and scaling to0..1 before
resampling. Those thresholds are stored CT values, not HU: with the saved
`HU=raw-1024` calibration, the clamp is[-944,-124]HU (midpoint-534, span820HU).
Thus `I=clip((HU+944)/820,0,1)`. Bone and other tissue above-124HU saturate
together, removing their internal intensity differences; boundaries against
lower-HU lung can still contribute. This is intensity saturation, not an
explicit bone-exclusion mask. Neither lung landmark TRE nor this code fact
establishes bone registration accuracy or isolates the benefit of LCC from
preprocessing. Prepared landmark coordinates obey the endpoint map
`native = crop_origin + reg * (crop_shape-1)/(reg_shape-1)`.

## Numerical method boundaries

The existing `lungct_no_sweep_repro/run_one.py` owns MR optimization unchanged.
The adapter supplies COPD64/32/16/8/4, downsample16/8/4/2/1,
400/400/400/400/900 steps, Adam1e-3, pTV-LCCsigma2.1mm and isoTV0.055,
csqrt0.005, border5, seed1, FP32 and exact knot insertion. An explicit
historical cuDNN benchmarkTrue/deterministicFalse override is recorded.

CorrField is the pinned standalone snapshot supplied through a read-only
configurable source root; its archive hash and actual module hashes are
checked/recorded. The existing torch.linalg.solve compatibility change is
retained. RawHU and fixed R231 masks are inputs; moving masks and landmarks
are excluded. All upstream scalar and two-stage parameters remain unchanged.
CorrField uses the default cuDNN benchmarkFalse/deterministicTrue.

CorrField receives `[1,1,X,Y,Z]` tensors. Its channel-last XYZ voxel field is
converted to `[Z,Y,X,3]` with ZYX components. Synthetic nonuniform-axis and
translation tests protect sign, layout and units. This is a pull field with
`warped_moving(x)=moving(x+u(x))` semantics.

## Evaluation and campaign gates

The canonical FP32 field retains the existing nominal1mm convention. Native
copies use the historical trilinear resize-to-native-crop and division by
native component spacing. Landmark positions use the actual endpoint map;
no new shape-rounding or displacement-scaling convention is introduced.

The shared evaluator reports native snap/subvoxel and reg1mm snap/subvoxel
TRE, per-landmark absolute native coordinates/displacements/predictions,
sample SDs and hashes. Native snap rounds field sampling and predicted points;
subvoxel uses trilinear sampling without prediction rounding. Crop and full
native-coordinate headline TRE must agree exactly. pTV reference fields are
native physical-mm XYZ fields and are evaluated using their saved landmarks
and spacing, independently reproducing saved TRE.

Preparation verifies all20 coordinate round trips and exact18-array parity
for each reused4DCT case. Reference replay verifies40 historical method/case
records. Preflight runs reduced COPD01/02 MR cases and full largest-volume
4DCT02/COPD10 CorrField cases, reusing the latter two in production.
The final queue contains30 unique new method/case records; no parameter
changes or result-driven reruns are selected after viewing quality.

## Reporting scope

Report order is COPD legacy/corrected/delta,4DCT three-method,COPD
three-method. Each case has300 paired landmark errors. Table entries use
within-case sample SD; the cohort row uses mean/sample SD over ten case
means. All70 evaluations and21,000 landmark metric rows are retained
externally, including historical references. Historical reuse and different
CorrField intensity/mask choices limit method-only and crop-only claims.

## Approved disconnected-graph recovery

The original sparse kNN graph can have multiple components; upstream MST/BFS
assumes every keypoint is reachable from node0 and otherwise indexes past the
available tree edges. Original4DCT04/05/06 attempts fail this way.

On2026-11-19 the user approved `forest_guard.py` for only those failed cases.
The wrapper forms no new edges and changes no costs or hyperparameters. It
runs the same upstream MST/TBP independently per component and retains unary
costs for isolated nodes. Connected graphs take the identical upstream tree
path; bitwise equality and exact component factorization are tested. This
extension is tagged rather than presented as unmodified upstream CorrField.

`guarded_runner.py` installs the guard only in its own process and attaches
all four stage/direction traces to the unchanged shared evaluator.
`recover_forest.py` freezes new source, retains failed attempts and partial
reports, and assembles30 accepted directories without replacing the27 earlier
successful fields. Final tables disclose the three guarded4DCT rows; all30
endpoints are independently verified from raw native landmarks.
