# Technical log: three-grid viewer

## Known limitations

- Coronal grid patterns hide AP displacement; row5 is a projected fixed-to-moving
  mesh and must not receive fixed-space contours.
- Reuse compact physical display displacement channels (audited <=0.1mm error),
  not original solver reevaluation. Raster border clamping differs intentionally
  from the analytic grid's repeating coordinates outside the source crop.
- Standard viewer moved to1101 in a separate completed update before this task.
  Do not restart or alter its retained8765 containers.

## 2026-11-19 preparation

- Read project workflows, protocol warnings, standard viewer/source and runtime.
- Port1102 free; standard1101 running. Existing13 display assets per case provide
  all needed images/masks and physical uL/uS/validity; no bundle changes needed.
- Largest checked nine-volume R8 cache is125.8MiB (COPD10), excluding one active
  replacement and existing display textures. Specification registered before work.

## Implementation and retained failures

- Added isolated grid-comparison application and optional server page selector;
  the standard application files were not edited by this task. Its preflight
  web hash snapshot differs from later files due to concurrent standard-viewer
  edits outside this task; preserve those changes and use the current1101 browser
  regression as evidence. No standard service was restarted.
- First synthetic-browser attempt used an intercepted document; Chromium local
  network policy blocked imports because the fake response lacked address-space
  metadata. Retained `validation/hardware_run1/`. Use a real HTTP document for
  synthetic GPU checks; no production CORS changes or browser bypasses.
- `hardware_run2` exposed case switching consuming stale/unapplied spacing text
  after programmatic rapid replacements. Fixed new-case selection to use the
  last applied spacing; explicit rebuild requests update the input. Failure and
  screenshot retained. Full acceptance rerun under `hardware_run3/` passed.

## Validation

- Twelve Python geometry/HTTP tests and three Node grid-model tests passed.
- Five synthetic GPU fields: zero, signed translation on anisotropic signed
  steps, scaling, nonlinear, invalid. Raster maximum coverage discrepancy is
  0.002119 or less, within1/255; identity/translation raster outputs exact.
  Grid centers/signs, opposite pull/push scaling, invalid omission and cancellation
  passed. GPU caches complete before Ready; nine R8 volumes retained.
- All20 immutable manifest receipts and260 compressed asset hashes match;
 6000 paired landmarks and guarded4DCT04--06 provenance preserved. Audit external
  `validation/unchanged_data.json`; no scientific arrays regenerated.
- Chromium141/NVIDIA RTX6000Ada hardware at1920x1080:4DCT01/02,COPD01/10
  image scrolling57.6--60.0fps, grid scrolling56.1--59.2fps, p95interaction
  16.4--21.6ms. Initial cache builds0.17--0.43s, replacement0.09--0.19s in
  this run. Cache62.7--125.8MiB. Timings are environment-specific.
- All four cases:8/12/8 spacing, rapid cancellation, busy slider/wheel gating,
  method order, light/dark, contour toggles, row5 overlay invariance, empty panel2,
 19 populated/six blank cells, zero post-Ready requests and no console errors.
  Repeated canceled downloads/cache builds/case switches retained24 GPU textures,
  one geometry and constant575098047-byte backing storage for repeated COPD10.
- Current1101 main viewer regression passed both image and landmark tabs,
  including CDF controls, method order, navigation, themes, cancellation/memory
  and request-free performance. Old8765 containers remain stopped and retained.
- Separate CPU-only service1102 uses the pinned existing image, artifact-owner UID/GID,
  read-only repo/artifact mounts, no GPU device requests and restart unless-stopped.
  No raw-data mount, registration rerun, source sync, staging or commit.
- Additional focused GPU checks passed: exact fixed-minus-fixed zero RGBA
  [237,233,224,255], rejection of fractional spacing without losing Ready,
  unapplied input discarded on case change, and actual guarded4DCT04 loading
  with visible provenance tag. External `validation/additional_checks.json`.
