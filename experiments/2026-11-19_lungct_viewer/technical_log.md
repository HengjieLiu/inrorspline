# Technical log

## Known limitations

- Pair-local anatomical LPS, not recovered scanner-world DICOM coordinates.
- Methods differ in optimization intensity/mask protocol; labels retained.
- Pull inversion may be ambiguous/unresolved; displayed explicitly.
- Browser timing depends on client renderer. Headless software performance
  cannot certify the user's desktop GPU.
- The earlier DIR-HNfibrosis example directory is absent on this host.

## Execution

- 2026-11-19: approved specification recorded before case preparation.
  P87 sources remain immutable; new versioned external viewer outputs only.
- Prepared all20 cases and260 compact assets, with300 correspondences each.
  Retained the first4DCT05 attempt: a single FP64 norm sum differed by
  8.9e-16mm under XYZ rather than the audited ZYX reduction order. Restoring
  that order reproduces every saved native TRE exactly; fields were unchanged.
- Nine geometry/inverse/warp/mask/HTTP tests passed. The compact-storage
  overflow warning is intentional in the FP32-fallback test.
- Independent smoke audits passed first9 and then12 cases, including COPD01.
  The complete20-case independent audit is the final data gate.
- Browser functional checks passed with zero console errors and no post-Ready
  requests. Earlier SwiftShader software timing was5.4fps and is retained;
  it does not meet the hardware target. Native NVIDIA RTX6000Ada OpenGL ES
  checks passed4DCT01/02 and COPD01/10 at49.0–60.0fps, p95interaction18.2–19.1ms.
  Linked cameras, point picking, method reorder/reload persistence, navigation,
  masks, zoom, canceled loading and disposal were exercised. All cases retained
  exactly14 GPU textures; collected ArrayBuffer storage was only3.4MiB above
  the current case's display arrays, with no preceding-case accumulation.
- Across18000 method/landmark inverse locations:17967 resolved,29 ambiguous,
  four unresolved. These remain labelled, with all300 points retained per case.
- A separate CPU-only read-only service runs on loopback8765; the P87 experiment
  container and all older runtimes remain preserved. Operational paths stay in
  ignored private configuration. No staging, commit, push or hosted publication.
- Final independent audit passed all20 cases/18000 method-landmark records.
  A reporting-only variable-shadowing bug initially labelled the complete
  audit partial; that report is retained under failed/, the variable was fixed,
  and the entire audit was rerun successfully. No case assets changed.
  Final acceptance seals the data/browser reports, exceptional-point CSV,
  pinned dependencies, screenshots and source snapshot externally.
- Access follow-up: user attempted the tunnel command on the data server itself.
  Added a separate read-only CPU service bound to its private network interface
  on port8765, retaining the existing loopback service. Direct root and catalog
  HTTP checks pass. No viewer code, prepared assets or scientific results changed;
  private interface and service details are recorded only in ignored configuration.
- UI revision R2: the user changed every difference reference to fixed, including
  moving-minus-fixed and fixed-minus-fixed. The selected-point crosshair and
  projected fixed-target ring retain their distinct coordinates; a visible legend
  and independent target-marker toggle clarify their roles. TRE readouts now use
  two decimals, without changing values or sorting. Added a persistent light/dark
  mode with unchanged scientific intensity/map colors. Reuses the20 sealed case
  bundles; new source snapshots and browser validation are versioned under
  `validation/ui_r2_fixed_theme/`. The original validation and source archive remain.
- R2 checks passed on the private-network origin: all five difference cells,
  GPU fixed-minus-fixed zero color, unchanged CT/map samples across themes,
  independent marker/contour controls, two-decimal readouts, theme reload
  persistence, request-free interaction and memory disposal. Four cases measured
  59.4–60.0fps with p95interaction20.2–20.6ms. A focused appearance check also
  verified the light-theme controls after correcting a CSS background shorthand.
  Rechecked all260 prepared asset checksums and20 case-manifest receipts unchanged.
- UI revision R3: added a second synchronized Previous/Next, landmark slider,
  position/ID/side and four-TRE readout immediately below the MPR image grid.
  Upper/lower controls share one selection, filtered range and wraparound;
  navigating below the images preserves the viewport. No prepared data changed.
  Focused browser validation and source snapshot are external under
  `validation/ui_r3_bottom_navigation/`.
- UI revision R4 authorized: add the four native-snap TRE step CDFs beside the
  two clouds, lung-filtered distributions and a selected-TRE sweep cursor with
  tie-inclusive counts. Full range is the initial view; local SVG reuses loaded
  records. Move the loopback and private-LAN services to1101, validate before
  stopping8765 services, and retain old containers. New evidence belongs under
  `validation/ui_r4_tre_cdf_port1101/`; sealed case bundles remain unchanged.
- R4 validation passed three Node CDF tests and nine Python geometry/HTTP tests.
  The first browser attempt passed the new chart checks but stopped on a legacy
  exact-equality camera assertion: a linked coordinate differed by2.8e-14.
  Retained that attempt under `failed_camera_roundoff/` and changed only the
  assertion to a tight1e-9 distance tolerance; no camera behavior was changed.
- The complete R4 Chromium141/RTX6000Ada rerun passed4DCT01/02 andCOPD01/10:
  scroll57.5–60.0fps, landmark navigation58.0–60.0fps, p95interaction19.7–22.0ms.
  Checked all side/metric/direction combinations, two navigation bars, cloud
  picking, tie counts, range/legend persistence, off-scale status and themes.
  No console errors or post-Ready requests;14 textures retained and preceding
  case ArrayBuffers released. Case cancellation, reorder/theme reload persistence
  and independent CT/map appearance checks also passed.
- Rehashed all20 manifests and260 assets against their existing receipts;
  all24000 TRE values including Initial are unchanged, with33 inverse exceptions
  retained. The new1101 loopback/private-LAN services passed page/catalog/asset
  and browser checks before stopping the two8765 service containers. Those old
  containers remain available for rollback. Source snapshot, acceptance receipt,
  browser screenshots/report and operational inspections are external; no
  scientific data, registration parameters or results changed.
- UI revision R5 authorized: move Initial/method TRE values from the two
  standalone strips into a second line beneath the landmark image-column
  titles. Preserve both navigation bars, two-decimal formatting, method
  identity on reorder, the CDF and graph-guard tags. Validation evidence is
  external under `validation/ui_r5_header_tre/`; prepared assets are reused.
- R5 browser checks passed on guarded4DCT04 andCOPD01: correct Initial/method
  subtitles, no Fixed subtitle or standalone strips, two-decimal audited values,
  both navigation bars, cloud picking, column reorder and dark/light rendering.
  All20 case manifests match R4. Screenshot review also exposed an older SVG
  visibility inheritance issue: a child intersection dot could remain visible
  after switching to anatomical/random order. Hiding the entire cursor group
  with display:none fixes that residue; final-source focused checks cover both
  the dot and off-scale arrow. No CDF values, scientific inputs or assets changed.
