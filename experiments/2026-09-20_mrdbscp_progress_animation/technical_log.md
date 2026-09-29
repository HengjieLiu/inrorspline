# Technical log: MR-D-BSCP presentation animation

## Known limitations / stop gate

- CP-BE has prior strict numerical-equivalence failures; no equivalence claim.
- First delivery is a CPU-only gallery of existing results, not a new experiment
  or a completed optimization animation. User case selection gates all capture.
- The frame slice is axial but all displayed quality metrics are full-volume.
- Actual host paths and generated assets are kept outside Git.

## Presentation revision audit and implementation

- User identified unequal initial grid spacing and fading/broken lines. Raw
  identity grid spacing was 8 voxels and initial displacement was zero, but
  nearest downsampling dropped 1-voxel strokes (6/7-pixel gaps mixed with 13).
  Trilinear sampling of the binary grid also mixed white stripes with black.
  Previous acceptance missed this final-panel rasterization defect.
- Approved grid-only geometric contour rendering with fixed white stroke width,
  plus independent every-step loss playback, in a NEW `render_v3/`. All previous
  render directories are retained; no numerical registration sources changed.
- New CPU regressions cover final-resolution line retention, source-coordinate
  translation sign/units, normal-motion invariance and independent loss cadence.
  Real clock/field snapshot timestamps remain explicitly distinguished.
- `render_v3/` passed nineteen CPU tests, fifteen render acceptance checks and
  eight independent checks. Encoded-frame comparison confirmed loss advances
  while the grid stays unchanged; frozen columns are pixel-identical.
- Final visual review found an endpoint-caption ambiguity introduced in v3:
  the field snapshot time was placed after "DVF file ready". Corrected the
  caption and added a regression; retain v3 and export final `render_v4/`.
  This is a text-only refinement, with identical grid and loss-playback logic.
- Final `render_v4/` completed: twenty CPU regressions, fifteen render acceptance
  checks and eight independent output checks passed. 900 decoded frames,
  1920x1080, 30fps, 30 seconds; all previous outputs and numerical inputs retained.
- CPU final-control reconstruction was exactly equal to the accepted GPU field;
  all 88 frozen numerical source hashes remained unchanged. No CUDA context was
  initialized. Existing measured timing and quality results are unchanged.
- Verified actual encoded loss motion between held-field frames, and exact
  pixel freezing of completed stage columns in all subsequent milestone PNGs.
  Inspected initial and final layouts: uniform initial grid, bright geometric
  contours, separate field-step/time caption and main warm clock, stage boundaries.
- Reproducibility/delivery records in the new external output include all source
  copies, invocation/environment/Git provenance, old/input/output hashes, grid
  coordinate arrays, every-video-frame mapping, tests, acceptance and report.
  Additional read-only frozen-source and notebook-output audits passed;
  `git diff --check` passed. No paper changes or commit.

## Implementation

- Read the reference demo and its axial orientation/rendering conventions.
- Added an independent source module with fixed configuration, full 3-D
  slice-grid pull sampling, gallery renderer, acceptance checks and unit tests.
- Reuse the accepted P2ad CP-BE-only warm outputs; verify each array hash against
  its existing evaluation before displaying the reused metrics.
- Existing Docker image and RO shared/RW artifact mounts verified. CPU plotting
  dependencies present; no GPU initialized by the dependency check.
- Preview sources do not modify the optimizer, existing acceleration backend,
  old outputs, PDFs or LaTeX.

## Verification / delivery

- Delivery 1 completed: `preview_20260920T221447Z_v1` beneath the experiment's
  external review root. Ten 1080p panels, ten slice archives, overview, offline
  HTML, case tables, readable report and reproducibility manifests generated.
- Ten CPU tests passed, including identity on all axes, in-plane translation,
  through-plane translation, nearest-label preservation, exact agreement with
  full 3-D warp on a small volume, grid orientation, checkpoint endpoints,
  first10 selection and output-directory safety.
- Independent delivery verification passed all thirteen checks: source/output
  hashes, source snapshots/no drift, count/order, finite slices/valid labels,
  PNG sizes, local HTML links/no remote dependencies, AST, tests, diff check,
  no CUDA initialization and expected gate status.
- Inspected rank-0 1080p panel and all-case overview visually: common orientation,
  readable labels/metrics and consistent MRI/segmentation/grid layouts.
- `unit_tests.txt`, `delivery_validation.json`, `docker_provenance.json`, exact
  replay commands, and verifier/environment-recorder sources are retained with
  the output. Renderer source was frozen at execution; post-render verifier
  additions are recorded separately rather than changing that snapshot.
- Docker inspection confirmed the existing image ID and artifact bind; no
  container changes. Entire preview was CPU-only; no registration launched.
- Stop gate active: await explicit rank/case choice before implementing and
  launching complete trajectory capture, offline per-snapshot Dice and animation.

## Case 04 continuation

- User selected Case 04 (rank 3, fixed 0117 / moving 0322); capture gate satisfied.
- Implementing opt-in baseline/clock/capture worker actions, separate monitored
  offline reconstruction and Dice, three-stage frozen-column renderer and player.
- GPU 0 checked idle before implementation (5 MiB, 0% utilization); recheck at
  launch through the existing continuous NVML monitor.
- Container lacked FFmpeg. A pinned external-only encoder dependency is used;
  no global package installation or Docker recreation is needed.
- Capture retains compact post-update controls every ten stage-local steps,
  initialization and actual final steps. Losses are intercepted after DVF-ready
  metadata begins; baseline logging never copies controls inside timed work.
- First smoke (`mrdbscp_progress_case04_20260920T223559Z_v1`) found an offline
  join bug: the legacy collector rewrites history `epoch` to global step, while
  callback events are stage-local. Registration losses and fields were exactly
  equal across all modes; error occurred after reconstruction during loss/time
  export. Fixed by explicitly joining `stage_epoch`, with a regression test.
  The failed smoke is retained. Formal timing had not started; use a fresh v2
  run/source manifest for the corrected workflow.

## Case 04 accepted delivery

- Corrected run `mrdbscp_progress_case04_20260920T223559Z_v2` completed. All
  fourteen accepted tasks (environment, smoke, excluded warmup, formal runs,
  separate evaluations) were clean; no contamination retries required.
- Formal sequence: one excluded full warmup, three baseline/clock pairs, one
  capture, then a fresh evaluation worker. Actual stage steps were 125/125/200.
- All formal loss rows, final fields, actual steps and stop categories matched
  exactly. Full-resolution reconstruction from the final saved controls also
  matched exactly. Forty-nine snapshots and all 450 step losses were retained.
- Selected the clock paired with the median baseline, as specified. Median
  paired clock overhead passed the 5% gate. Full timings, including the slower
  first baseline, remain in the external report; no measured times were adjusted.
- Presentation delivery: `render_v2/` under the matching external review run.
  MP4 decoded successfully as 1920x1080 H.264/yuv420p, 30 fps, 900 frames/30 s.
  Offline HTML, all 51 presentation-state PNGs, stage-final PNGs, CSV/JSON,
  acceptance, exact render invocation and source snapshot retained.
- `render_v1` is preserved. The presentation-only v2 adjustment adds room after
  step 450 and avoids drawing the cursor over the final dashed stage boundary;
  no optimization or evaluation was rerun for this visual refinement.
- Fourteen CPU contracts and fifteen independent delivery checks passed.
  Inspected final and stage-2-final layouts, including frozen columns and
  loss boundaries. HTML assets resolve locally; no CDN dependency.
- Final DVF array SHA256 exactly matched the historical accepted CP-BE Case 04
  evaluation. Its HD95/Jacobian metrics were reused explicitly after that check;
  fresh final Dice matched the stored metric. No metric work entered timing.
- Full source/data/Git/Docker provenance and GPU monitoring remain in the run;
  renderer/encoder hashes, output hashes and detailed report remain in review.
