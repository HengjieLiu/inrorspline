# MR-D-BSCP presentation progress visualization

## Fixed 50/50/50 cache/batch presentation profile

The `fixed50_cached_batch` profile is an explicit presentation-only variant of
the Case 04 workflow. It runs exactly 50 optimizer updates at each of the three
8/4/2 control-spacing stages, enables only `cache,batch,cpbe`, and disables
adaptive early stopping. The historical adaptive CP-BE profile remains the
default when `--profile` is omitted.

```bash
python3 -B -m scripts.visualization.mrdbscp_progress.workflow \
  --container "$CONTAINER" --repo-container "$REPO_CONTAINER" \
  --artifact-host "$ARTIFACT_ROOT" --artifact-container "$ARTIFACT_CONTAINER" \
  --oasis-root "$OASIS_ROOT" --run-id "$NEW_RUN_ID" \
  --profile fixed50_cached_batch --phase run
```

After accepted capture/evaluation, render into a new directory with:

```bash
python -B -m scripts.visualization.mrdbscp_progress.render_compact \
  --data "$DATA_JSON" --output "$NEW_RENDER_ROOT"
python -B -m scripts.visualization.mrdbscp_progress.verify_compact \
  --render-root "$NEW_RENDER_ROOT"
```

The compact 16:9 layout places enlarged, tightly spaced image/label/grid panels
on the left and one combined loss panel on the right. It plots measured raw
per-step values in three colors. Because the optimizer minimizes
`-NCC + 1000*BE`, the display adds the same constant one to similarity and total
to show `1-NCC`, `1000*BE`, and `(1-NCC)+1000*BE`; this does not alter the
optimization, gradients, stage schedule, or recorded stopping behavior. The
title and annotation fonts are enlarged for projected slides.

The final compact renderer produces a 15-second, 30-fps replay. It assigns 12
playback seconds to optimization in proportion to the three accepted measured
stage spans. Preparation, stage transitions, final GPU-to-CPU/NumPy output and
holds use short explicit phases with visible status text. Playback scaling is
therefore intentionally nonuniform. Historical timing contains stage totals,
not clean per-step spans, so steps are evenly placed inside each stage; field
panels still change only at actual saved ten-step states. The tiny aggregate
preparation residual is split equally for display and is not a claim of three
separately measured setup durations. `presentation_timing.json` records every
allocation and `frame_timeline.csv/json` maps all 450 frames.

When an accepted Case04 fixed-50 timing/field record already exists, the
capture-only path avoids a timing rerun:

```bash
python3 -B -m scripts.visualization.mrdbscp_progress.capture_historical \
  --container "$CONTAINER" --repo-container "$REPO_CONTAINER" \
  --artifact-host "$ARTIFACT_ROOT" --artifact-container "$ARTIFACT_CONTAINER" \
  --oasis-root "$OASIS_ROOT" --run-id "$NEW_RUN_ID" \
  --historical-review-host "$HISTORICAL_REVIEW" \
  --historical-run-container "$HISTORICAL_RUN_CONTAINER"
```

This opt-in path monitors but does not wait for an otherwise busy GPU0 because
the snapshot run is explicitly excluded from timing. It requires the complete
150-step loss trajectory and final DVF to match the accepted historical record,
records both file and array hashes, and reconstructs metrics separately. The
video keeps the concise `Warm elapsed` text; the historical timing source and
phase-specific playback mapping are disclosed in machine provenance and the
report.

### Dense measured-field variants

Use the `fixed50_cached_batch_dense_capture` profile to save every post-update
control state once. This capture is not a timing run. After exact trajectory
and final-DVF matching, the same 153-state evaluation can produce multiple
measured cadences without rerunning registration:

```bash
python -B -m scripts.visualization.mrdbscp_progress.render_compact \
  --data "$DENSE_DATA_JSON" --output "$NEW_STEP5_ROOT" \
  --field-step-interval 5 --video-name animation_step5.mp4
python -B -m scripts.visualization.mrdbscp_progress.render_compact \
  --data "$DENSE_DATA_JSON" --output "$NEW_STEP2_ROOT" \
  --field-step-interval 2 --video-name animation_step2.mp4
python -B -m scripts.visualization.mrdbscp_progress.render_compact \
  --data "$DENSE_DATA_JSON" --output "$NEW_STEP1_ROOT" \
  --field-step-interval 1 --video-name animation_step1.mp4
```

These variants display only `Stage n/3 · Optimizing · Step x/50` and the final
`DVF file ready` state. The accepted historical endpoint and stage spans remain
the elapsed-time source. Setup, transition and output phases are omitted from
the visual playback, not remeasured or subtracted from the accepted endpoint.

## Revised grid and independent loss playback

For an already accepted Case 04 render, use the new CPU-only revision command:

```bash
python -B -m scripts.visualization.mrdbscp_progress.render_revision \
  --data "$DATA_JSON" --previous-render "$PREVIOUS_RENDER_ROOT" \
  --output "$NEW_RENDER_ROOT"
python -B -m scripts.visualization.mrdbscp_progress.verify_revision \
  --render-root "$NEW_RENDER_ROOT"
```

Use the Docker/encoder environment below, with CUDA hidden. The output MUST NOT
exist. The command runs both CPU test suites, reconstructs coordinate maps from
49 saved controls, checks final-DVF tolerance and old-output/input hashes, and
renders a new MP4/player/report without optimization or metric evaluation.
The old generic finalizer is not needed; this revision has its own acceptance.

Grid lines are level sets of full-resolution pull coordinates at source-voxel
multiples of 8, with fixed 1.2-point white strokes at final display resolution.
This replaces the binary-texture grid only in the revision, not historical
previews. The conceptual reference lattice extends beyond the source FOV and
through-plane; viewport clipping remains. MRI/label sampling is unchanged.
Antialiased stroke edges can be gray; lines are no longer faded by intensity
interpolation. Compressed/folded geometry can still produce overlapping contours.

Loss traces reveal the real every-step records at 30fps against the measured
warm timeline; images/labels/grid/Dice hold the last saved state. No smoothed
losses or invented fields. Caption gives FIELD snapshot step/time; the main
clock follows the slowed recorded interval and freezes on explicit holds.
`frame_timeline.csv/json` records this distinction for all 900 video frames.
The offline HTML plays the actual video and supports snapshot and stage jumps.

Dedicated source-only module for an English 16:9, 1080p presentation of
`MR_A_fast_decay + cpbe`. No cache/batch/NCC-cache/loss-buffer features.
Generated material belongs under `${ARTIFACT_ROOT}`, never this directory.

## Delivery 1: case-selection gate

Implemented: first10 accepted-field preview, fixed presentation configuration,
3-D pull-warp slice sampling, offline gallery, source/data provenance, validation
and CPU tests. **No registration or trajectory capture is launched by preview.**
The user selected Case 04 (seed-3 rank 3, fixed 0117 / moving 0322). Delivery 2
below implements the approved selected-case capture and animation workflow.

Run in the established Docker environment, as the artifact owner's UID/GID.
Provide mounted paths through local configuration. Use `CUDA_VISIBLE_DEVICES=''`
for this CPU-only phase. Set `MPLCONFIGDIR` to a writable external cache location.

```bash
python -B -m unittest scripts.visualization.mrdbscp_progress.test_progress -v
python -B -m scripts.visualization.mrdbscp_progress.preview \
  --source-review "$SOURCE_REVIEW" \
  --source-run "$SOURCE_RUN" \
  --oasis-root "$OASIS_ROOT" \
  --pair-file configs/splits/oasis/pair_test_200_shuffled_seed000003.txt \
  --output "$ARTIFACT_ROOT/repo_experiments/$EXPERIMENT_KEY/$RUN_ID/preview"
```

The output directory must be new. Existing results are never overwritten.
The input is the completed P2ad experiment: ten `headline`, `cpbe`, `warm`
records, ranks 0–9, with a passing acceptance manifest. DVF array hashes must
match the corresponding full-volume evaluation records. Clean status, backend
flags, feature selection, pair IDs, shape/dtype/finiteness and raw geometry are
checked before use. The external manifest records source hashes and copies,
Git commit/dirty status/patch, runtime, raw input checksums and exact Python argv.

Open `index.html` locally; no server or CDN is needed. It contains the first10
overview and ten 1920×1080 panels. Reply with a **rank 0–9** (the gallery also
shows one-based case numbers). The page does not launch jobs or submit choices.

### Visual conventions

- Match the reference registration demo's visual vocabulary: grayscale MRI,
  fixed named-color label palette, white pull-warped grid on black. Renderer and
  geometry are independently implemented, without importing another checkout.
- Default axial plane is reoriented array axis 2, index 96 (zero-based), not the
  PyTorch D axis. Raw LIA input orientation/spacing/affine are verified.
- Transpose, horizontal flip, `origin=lower`: left R, right L, top A, bottom P.
- Original pairwise min-max normalization; shared display window `[0, .75]`.
- Warp full 3-D source volumes at the slice coordinates, retaining through-plane
  displacement. `align_corners=True`, zero padding, trilinear MRI/grid and
  nearest-neighbor segmentation. This is not a 2-D-only registration display.
- Reference grid spacing is 8 voxels, independent of the method's control grid.
- Numbers are the existing whole-volume 35-label Dice/HD95 and foreground
  Jacobian metrics, not slice-specific metrics. The display does not recompute
  registration metrics. Warm time is the prior measured per-case endpoint time.
- CP-BE's strict numerical-equivalence failure remains a known limitation.

## Delivery 2: selected Case 04

The approved design lives in the experiment execution specification. It retains
Moving/Fixed columns and three sequentially frozen stage columns; separate
per-step similarity, weighted BE and total-loss strips; post-update snapshots
every ten local steps plus initialization and the actual stage endpoint; offline
full-volume Dice; a separately measured warm timeline; 30-second slowed replay;
MP4, offline player, PNGs, CSV/JSON, and readable report. Do not use a snapshot
run's wall time or the ten-case mean as the displayed case's real warm time.

### Capture and evaluate (host orchestration, Docker workers)

```bash
python3 -B -m scripts.visualization.mrdbscp_progress.workflow \
  --container "$CONTAINER" --repo-container "$REPO_CONTAINER" \
  --artifact-host "$ARTIFACT_ROOT" --artifact-container "$ARTIFACT_CONTAINER" \
  --oasis-root "$OASIS_ROOT" \
  --run-id mrdbscp_progress_case04_<timestamp> --phase dry-run
```

Repeat with `--phase run`. The selected-case workflow intentionally fixes rank 3
and physical GPU 0. It runs CPU contracts and environment checks, a baseline /
clock / capture one-step smoke and separate smoke reconstruction, then one
excluded full warmup and three paired warm baseline/clock runs plus one snapshot
run. A separate worker evaluates all saved states only afterward. Continuous
NVML monitoring rejects contaminated attempts. It never kills other GPU jobs.

Every baseline retains every loss row after the DVF endpoint. The clock callback
records only host timestamps after existing scalar reads; capture copies controls
and is **not** a timing result. Match the loss table using `stage_epoch`, not
`epoch` (the legacy collector makes `epoch` global). A regression test covers this.
The median-baseline pair selects the displayed clock; absolute median paired
clock overhead must be <=5%. No scaling/subtraction of measured seconds.

The failed initial smoke is retained separately; always use a NEW run ID when
changing capture/numerical sources. `formal_source_sha256.json` freezes those
files, and `progress_source/` plus normal worker provenance retain source copies.
Rendering-only changes do not alter accepted timing sources.

### Encode (CPU inside Docker, after GPU jobs)

Use pinned `imageio-ffmpeg==0.6.0` installed with `pip --target "$VIDEO_TOOLS"`
under external artifact storage, never globally. Add that directory to the
container's `PYTHONPATH` when rendering. `DATA_JSON` is the `animation_data`
path in the completed run's `formal_evaluation.json`; `RENDER_ROOT` must be new.

```bash
python -B -m scripts.visualization.mrdbscp_progress.render \
  --data "$DATA_JSON" --output "$RENDER_ROOT"
python -B -m scripts.visualization.mrdbscp_progress.finalize \
  --run-root "$RUN_ROOT" --render-root "$RENDER_ROOT" \
  --historical-review "$SOURCE_REVIEW"
```

Deliverables: 900-frame H.264/yuv420p MP4, no-CDN offline HTML with play/pause,
speed/scrub/stage jumps, every rendered PNG, stage-end/final PNGs, every-step
losses, snapshot/per-label Dice, measured timeline, playback mapping, report and
acceptance. All stages show full-resolution reconstructions; no invented
intermediate deformations. Six seconds of explicit holds plus uniform time
scaling produce the 30-second video. Report separately the clean baseline,
measured clock, and slower capture duration. Keep the control archives in the run
for subsequent slicing/re-rendering. Final HD95/Jacobian metrics are transferred
from historical evaluation only when case and DVF array bytes match exactly.

The renderer decodes the exported video to verify size, FPS, codec/pixel format
and frame count. The finalizer checks clean monitoring, frozen sources, mode
order, field/loss parity, stage freezing, local assets and endpoint semantics.
Human review of the PNGs/video remains necessary before using it in a talk.
