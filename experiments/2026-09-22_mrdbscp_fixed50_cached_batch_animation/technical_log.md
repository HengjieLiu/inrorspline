# Fixed-50 cache/batch animation technical log

## Known limitations

- Single selected case; no cohort-performance claim.
- CP-BE strict FP32/Adam equivalence warning remains.
- Loss rows are pre-update while saved controls/fields are post-update.
- Displaying1-NCC and shifted total changes only the plotted offset.
- Ten-step field snapshots are held between measurements; no field interpolation.

## Implementation record

- Historical adaptive CP-BE profile remains the default.
- Added an explicit fixed50 cache/batch profile and generalized capture/final
  validation against recorded profile metadata.
- New rendering is isolated from historical render directories and uses the
  accepted geometric-grid convention.

## Completed execution

- Accepted run: `mrdbscp_progress_case04_fixed50_20260923T021216Z_v2`.
- The original clean-timing queue was stopped after the user authorized reuse
  of accepted historical timing. No new timing benchmark was performed.
- Reused Case04 warm endpoint: `1.8939262107014656 s`; stage spans:
  `0.5220975875854492`, `0.5528466701507568`, `0.6345169544219971 s`.
- The GPU0 capture overlapped an unrelated monitored job and took
  `23.53174851089716 s`; this value is retained as provenance and excluded from
  the display clock and performance claims.
- Capture produced 150 exact-matching loss rows and 18 compact-control states.
  Its final `.npy` file and array hashes matched the accepted historical field;
  reconstructed final-field RMS/max errors were both zero.
- `render_v1` exposed title, axis-note and Dice-label overlap. `render_v2`
  removed title/Dice overlap but retained one loss-axis annotation collision.
  Both are preserved. `render_v3` fixed the layout; `render_v4` additionally
  removed paired-clock language that did not apply to historical timing.
- `render_v4`: 1920x1080, 30 fps, 30 s, 900 frames, H.264/yuv420p. Renderer,
  independent decoded-frame verifier and final delivery audit all passed.
- The 30-second playback audit found that capture-derived within-stage event
  fractions made the visible loss progress highly nonuniform and created long
  apparent freezes. These were presentation-clock artifacts, not measured
  preparation time. `render_v5` introduced a 15-second phase-aware clock and
  explicit preparation/optimization/output status; it is retained because one
  long loss-value line exceeded the right edge.
- Final `render_v6`: 1920x1080, 30 fps, 15 s, 450 frames, H.264/yuv420p. It
  allocates 12 playback seconds to optimization in proportion to accepted
  stage spans and short explicit intervals to setup, transitions and output.
  Steps are evenly positioned within each stage because historical timing only
  measured stage totals. The aggregate preparation residual is split equally
  for display and is not claimed as three independently measured durations.
  Status text distinguishes preparation, optimization, stage completion and
  DVF serialization; step/loss values sit in a two-line band above the plot.
- `render_v6` passed 23 renderer contracts, codec/frame/duration checks,
  independent decoded-frame motion/static-region checks, pixel-identical
  frozen-column checks, source-hash checks and final delivery validation.
- Exact-hash quality reuse: Dice `0.8302474168`, HD95 `1.7007920447 mm`,
  foreground SDlogJ `0.8360148396`, foreground folding `4.2021090762%`.

## Dense-cadence extension

- Requested variants: measured field/image/label updates every 5, 2 and 1
  optimizer step(s), with unchanged historical warm elapsed time.
- Existing captures contain only ten-step controls, so they cannot support the
  requested variants without inventing fields. One new every-step control
  capture is required; it is explicitly excluded from timing.
- GPU0 preflight on 2026-09-22 found no compute process and 5 MiB/49140 MiB in
  use. The three videos will share this one dense numerical capture.
- Accepted dense run: `mrdbscp_progress_case04_dense_step1_20260923T044548Z`.
  The every-step capture was monitor-clean, stored 153 states, reproduced all
  150 historical losses and matched the final DVF exactly. Its 16.578717 s
  wall time is excluded from all displayed timing.
- The first `step5/` render stopped before encoding because the separate final
  state had no frame after transition phases were removed. It is retained as a
  failed audit. The renderer now uses the final measured field directly for
  the DVF-ready hold; accepted step-5 output is `step5_v2/`.
- Accepted outputs: `step5_v2/animation_step5.mp4`,
  `step2/animation_step2.mp4`, and `step1/animation_step1.mp4`. Each is
  1920x1080, 30 fps, 15 s, 450 frames, H.264/yuv420p. Their measured field
  counts are 33, 78 and 153 respectively; no field interpolation is used.
- All three passed 23 renderer tests, exact cadence checks, decoded-frame
  checks, pixel-identical frozen-stage checks, exact final-DVF/quality reuse,
  provenance checks and final delivery validation.
