# Case04 fixed-50 cache/batch CP-BE presentation animation

## Objective and boundaries

Create a new, non-overwriting presentation animation for seed-3 OASIS Case04
(rank3, fixed0117/moving0322) using the fastest screened schedule:
`50/50/50 + cache,batch,cpbe`. Preserve the accepted pull-warp, normalized XYZ,
full-resolution output, alpha1000, seed1, float32, stage LR .03/.01/.003,
spacing8/4/2, image downsampling4/2/1, exact knot insertion, and cuDNN
benchmark=False/deterministic=True. No paper, PDF, LaTeX, production default or
historical artifact is modified.

This is a single-case teaching visualization, not cohort evidence or proof that
CP-BE is numerically equivalent to dense analytic BE. Segmentation and Dice are
offline-only and never enter optimization.

## Capture and displayed-time protocol

The initial plan was a clean sequential GPU0 rerun with three paired timing
measurements. During execution, the user explicitly replaced that requirement:
do not rerun timing; reuse the accepted historical Case04 warm timing and run
only the snapshot capture in remaining GPU0 memory. The accepted timing source
is P2af headline row
`headline_fixed_50_50_50_cached_batch_004_attempt0`: endpoint `1.8939262107 s`
and stage loop/postloop spans `0.5220975876`, `0.5528466702`, and
`0.6345169544 s`. The video itself retains the concise `Warm elapsed` label;
the historical source is recorded in report/provenance rather than on-screen.

The concurrent capture is numerical input only. Its wall time is excluded, and
GPU monitoring records the unrelated process rather than treating the capture
as a timing result. The final presentation uses only accepted endpoint and
stage totals: optimization steps are evenly positioned within each measured
stage span, rather than inheriting contaminated capture-time fractions. The
small measured endpoint residual before the in-memory DVF is split equally
across three displayed preparation phases because endpoint timing cannot
separate those calls. Final GPU-to-CPU/output time comes from the accepted
endpoint marks. This mapped clock is explanatory playback, not a new per-step
timing benchmark.

Save initialization, every ten local steps and each stage endpoint. All stages
end at50, producing six states per stage before cross-stage deduplication rules.
Save every actual pre-update loss row. Offline, reconstruct every control state
at full resolution, compute 35-label whole-volume Dice, and verify the final
field against the captured DVF. Reuse final HD95/regularity only after exact
array-hash matching to the completed first10 fixed-50 cache/batch experiment.

## Presentation layout and loss semantics

Render 1920x1080,30fps,H.264/yuv420p, 15 seconds. Preserve the
geometric white pull-coordinate grid from the accepted Case04 revision. The
left side contains tightly spaced Moving, Fixed and three sequentially frozen
stage columns across MRI, segmentation and grid rows. The right side contains
one combined loss panel with raw per-step curves and no cross-stage smoothing:

```text
similarity_display = 1 - NCC = 1 + data_loss
weighted_BE        = 1000 * reg_loss
total_display      = (1 - NCC) + weighted_BE = 1 + optimized_loss
```

The +1 shift is display-only and does not change gradients, updates, stopping
or fields. Use distinct colors, stage-boundary dashed lines, a moving cursor,
larger presentation fonts and continuous every-step loss reveal while fields
remain at measured ten-step snapshots. Allocate 12 playback seconds to
optimization proportionally to the three accepted stage spans; compress setup,
stage transitions and output, and show their status explicitly. Keep the real
measured warm clock while the playback itself is intentionally nonuniform.

## Acceptance and outputs

Require accepted clean historical timing, recorded concurrent-capture
contamination, correct features and backend flags,150 matching loss rows,
expected snapshots, finite
float32 final DVF, reconstruction tolerance RMS<=.005 voxel/max<=.05 voxel,
monotonic mapped timing, exact historical/capture file and array hashes, frozen
completed-stage columns, readable
1080p text, valid450-frame encoding, local HTML assets and recorded source/data
hashes. Keep machine-readable timelines, losses, snapshots, Dice, provenance,
commands, reports, controls and validation under `${ARTIFACT_ROOT}`.

Final accepted run: `mrdbscp_progress_case04_fixed50_20260923T021216Z_v2`.
Final accepted presentation output: `render_v6`; `render_v1` through `render_v5`
remain preserved visual-layout/report audits.

## Dense field-update extension

Create one new non-timing GPU0 capture with post-update controls saved at every
step. Require the 150-step loss trajectory and final DVF file/array hashes to
match the accepted historical Case04 result exactly. Reconstruct all 153
stage-local states (`0..50` for each stage) once, including whole-volume Dice
and central-slice MRI/label/grid artifacts.

Render three independent, non-overwriting 15-second outputs by selecting only
measured states at intervals 5, 2 and 1. Name the MP4s
`animation_step5.mp4`, `animation_step2.mp4`, and `animation_step1.mp4`.
Do not interpolate deformation fields. Preserve the accepted historical warm
endpoint `1.8939262107 s` and stage spans. The visual timeline contains only
the three `Optimizing` stage/step phases followed by `DVF file ready`; setup,
transition and serialization phases are omitted from presentation playback.
Their omission must not alter the recorded endpoint or optimization data.
