# MR-D-BSCP fast progress animation

## Approved presentation-only revision

Preserve `render_v1/` and `render_v2/`; produce a NEW revision from the
accepted Case 04 animation data. Initial revision `render_v3/` is retained;
final `render_v4/` additionally clarifies field-snapshot versus file-ready
timestamps after visual review. No optimization, timing or metric rerun.
CPU-only Docker reconstruction uses saved controls to recover full-resolution
pull coordinate maps at axial index 96. Draw source-voxel-coordinate contours
at multiples of 8, white fixed-screen-width antialiased strokes, instead of
trilinearly warped binary grid textures. The reference lattice extends through
and beyond the image volume; only display-viewport clipping applies. MRI and
label warping/padding are untouched. Test identity spacing at the actual small
panel resolution, pull translation, through-plane invariance and final-DVF parity.

Decouple loss playback from 10-step field snapshots: reveal each recorded loss
when its measured clock timestamp is reached by 30fps playback. No loss
smoothing or synthetic deformation interpolation; stage curves stay separate.
Main warm clock follows uniformly slowed recorded time and freezes on holds;
caption identifies the held field snapshot's local step and timestamp.
Retain 49 field states and all 450 loss points, with an additional per-video-frame
CSV/JSON mapping. HTML uses the same MP4 with speed/scrub/stage-jump controls.
Hash old outputs and numerical inputs before/after; new source snapshots,
commands, tests, acceptance and report remain in the new external directory.

## Status and execution gates

Approved two-delivery workflow. Delivery 1 builds first10 static previews from
existing accepted CP-BE warm results, then **stops for user case selection**.
Delivery 2 implements and executes the selected-case trajectory and animation.
Never start registration before a case has been selected. No paper edits,
changes to production method defaults, or commits are included.

Known limitations: single-case teaching visualization, not a cohort performance
claim; CP-BE is not numerically equivalent to the original FP32 BE/Adam path.
Metrics are full-volume rather than central-slice measurements. Slowed playback
is not optimization wall time. Existing unrelated/previous edits are preserved.

## Delivery 1: static selection gallery

- Source: accepted P2ad, `mrdbscp842_be_combo_20260920T022500Z_v1`, only ten
  headline CP-BE warm records in seed-3 rank order. Do not rerun registration.
- Validate accepted manifest, clean rows, features `cpbe` only, F/T backend,
  field hashes against evaluation, data/pair identity and input spatial geometry.
- CPU-only processing in existing Docker, artifact-owner UID/GID, raw data RO.
  GPU 0 is reserved for future registration, not used by this preview.
- Create ten 1080p PNG panels, a ten-case overview, a standalone HTML gallery,
  slice archives, case CSV/JSON, validation, source/input provenance and report.
- Default reoriented axial axis 2/index 96. Identical display convention for
  MRI, labels and grid; reference-grid spacing 8 voxels. Full 3-D pull sampling.
- Case selection is explicit; no ranking or automated best-case selection.

Source module: `scripts/visualization/mrdbscp_progress/`.
External preview/review root:
`${ARTIFACT_ROOT}/repo_experiments/2026-09-20_mrdbscp_progress_animation/<run-id>/`.
Future registration root: `${ARTIFACT_ROOT}/runs/mrdbscp_progress_<timestamp>/`.
Host/container paths remain in private configuration or external manifests.

## Delivery 2: selected-case capture and real warm clock

User selected **Case 04 / seed-3 rank 3, fixed 0117 / moving 0322** after
reviewing the gallery. The case-selection gate is satisfied. Use a new run ID
`mrdbscp_progress_case04_<timestamp>` and preserve the preview and prior results.
One-step smoke covers baseline, clock, capture and offline reconstruction/Dice
before the full sequence. The worker only gains opt-in progress actions;
existing registration commands/defaults and the optimizer remain unchanged.
Video encoding uses pinned imageio-ffmpeg 0.6.0 in an external tools directory,
not a global container install. The original Docker image is retained.

Frozen method: `MR_A_fast_decay + cpbe`, alpha 1000, seed 1, float32, six CPU
threads, physical GPU 0, benchmark False/deterministic True; remaining flags
match P2ad. cps 8/4/2, downsampling 4/2/1, LR .03/.01/.003, maximum epochs
400/800/1300, unchanged adaptive stopping, absolute-field initialization and
exact knot insertion. No cache/batch/NCC caching/loss buffering features.

Reuse the existing post-update control callback. Save compact controls and
geometry at initialization, each ten stage-local steps, and actual stage ends;
deduplicate endpoints while retaining final-state semantics. Save every loss
row, stage/global epoch, geometry, stopping reason and provenance. Training
losses are pre-update; saved controls are post-update. No mislabeled re-evaluation.

Within one persistent warm worker: one excluded complete warmup, three paired
endpoint-only baseline/time-axis runs, then capture. Select the time-axis run
paired with the median baseline time, never the fastest. Clock callbacks record
monotonic host timestamps after existing scalar reads, without extra per-step
CUDA synchronization or tensor copies. Origin is `case_load_complete`, endpoint
`dvf_file_ready`, including stage preparation/transitions and output writing.
Retain timing source semantics; do not adjust measurements to match a 5.59s mean.
If median paired clock overhead exceeds 5% in absolute value, investigate before
publishing the animation. Snapshot/evaluation/render durations stay separate.

Require matching stage counts/stopping categories/loss trajectories; report
exact field matches preferentially, otherwise existing RMS <=.005 voxel and
maximum <=.05 voxel limits. Failed parity prohibits joining snapshot states to
the measured clock. Monitor GPU 0 approximately once per second plus boundaries;
retain contaminated attempts and retry after clearance without stopping others.

## Offline reconstruction and layout

Reconstruct each control snapshot at full (160,224,192) resolution using full cps,
not stage domain cps. Same normalized XYZ pull convention and interpolation as
the method. Nearest labels; existing 35-label Dice and missing-label rules for
every snapshot. No intermediate HD95/Jacobian work by default. No need to retain
every full DVF; keep controls, final field, slice arrays and per-label metrics.

English 1920x1080: Moving, Fixed, Stage 1, Stage 2, Stage 3 columns; MRI, labels,
pull-grid rows; stage Dice beneath. Completed stage columns freeze. Titles show
cps/downsampling, active stage/local steps and measured warm seconds. Bottom has
three aligned raw per-step similarity=-NCC, weighted BE=1000*BE, total curves.
Stage boundaries become dashed lines on completion; no interpolation/smoothing
across the change of loss resolution. Record raw unweighted BE too.

Default 30-second playback at 30fps: 1s initial hold, 1s after stages 1 and 2,
3s final hold, remainder uniformly scaled from actual time. Hold the last true
snapshot between sampled updates; no invented intermediate deformations. Holds
freeze the displayed actual clock. Label slowed replay/offline metrics.

## Acceptance and handoff

Run CPU geometry/selection/output-safety tests first. Validate identity,
known translation (including through-plane), nearest labels, grid direction,
snapshot endpoints, timing monotonicity, frozen columns and trajectory parity.
Final field must be finite float32 normalized XYZ (160,224,192,3); reconstructed
final snapshot must match it. Manually inspect orientations/text at 1080p.

Delivery 1 ends with verified gallery and user case selection. Delivery 2 ends
with H.264/yuv420p MP4, offline playback/scrubbing/speed/stage-jump HTML, stage-end
PNGs, loss/time/snapshot/Dice/playback CSV/JSON, acceptance and Markdown report,
source/data/environment hashes, Git dirty provenance, commands and monitor logs.
Changing playback formatting needs only rendering; a different slice can be
reconstructed from controls without rerunning registration.
