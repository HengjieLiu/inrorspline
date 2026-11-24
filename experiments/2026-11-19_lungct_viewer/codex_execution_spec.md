# DIR-Lab private case viewer

Approved 2026-11-19: all20 P87 4DCT/COPD cases, one loaded case, two tabs,
method order MR-D-BSCP/pTVReg/CorrField. Posthoc only; no registration changes.

## Inputs and geometry

Resolve P87 accepted campaign/reference manifests; corrected MR, historical
pTV and accepted CorrField including tagged forest-guard4DCT04–06. Raw CT,
R231 left/right labels and fields are read-only. RawZYX is displayed in
pair-local LPS: X towardL, Y towardP, rawZ towardI. Use explicit crop origins
and endpoint-aligned physical spacing, not assumed exact1mm affines.
Pull fields map fixed to moving. Invert for moving-landmark display only;
native snap TRE is copied/replayed independently, never replaced by inversion.

## Implementation

Source: scripts/visualization/lungct_viewer. Buildless JS/Three.js/WebGL2,
shared renderer, Python read-only loopback HTTP server, pinned local assets.
Generated bundles/provenance/tests/screenshots/dependencies live under
${ARTIFACT_ROOT}/prepared/lungct_viewer/<version>. Preparation runs inside
the established pinned P87 Docker runtime with CUDA hidden. Native shell
downloads and Node browser testing use separate external dependencies.

Comparison:5x5 coronal layout, shared scrolling/windowing, signed differences,
pull grids, magnitude and signedSI with shared symmetric scale. Cyan anatomy
and yellow fixed contours. Landmark:2 selectable/linked3D point clouds then
3x5 orthogonal images, original and inverse-mapped centers, fixed targets,
native TRE sorting, superior sorting, random, side filtering, wrap navigation,
crosshair toggle and full/10/20/50mm extents.

All work required for both tabs finishes before Ready. No network requests
or image/field reconstruction after Ready. Release old case on switching.
Finite compact display arrays must remain within2HU/0.1mm of FP32 source;
otherwise useFP32. Sources unchanged. Inverse tolerance0.05mm; ambiguous and
unresolved cases explicit. Blank cells stay blank; valid field domains retained.

## Acceptance

Synthetic geometry/translation/inverse/mask/difference/zoom tests;20 complete
bundles,300 points each, native TRE replay, source hashes. Browser QA on
4DCT01/02 and COPD01/10:controls, request-free Ready state, cancellation,
memory disposal, orientation and signed scales. Target60fps, gate30fps and
p95interaction<100ms on recorded browser environment; report software-rendered
headless timing separately from hardware desktop performance if applicable.
No stage/commit/push or public deployment.

## User UI revision R2

Use fixed as the difference reference in all five columns: moving-minus-fixed,
three deformed-minus-fixed cells and fixed-minus-fixed. Keep the selected-point
crosshair and projected fixed-target ring distinct; explain them visibly and
make the target ring independently switchable. Format all four landmark TRE
readouts to two decimals, preserving stored/sort precision. Add persistent
dark/light presentation without changing CT values or map colors. Reuse the
sealed20-case data and retain the original source/validation snapshots; validate
the revision separately under external validation/ui_r2_fixed_theme/.

## User UI revision R4: TRE cumulative curve and port1101

Add a third panel alongside Moving/Fixed clouds. Reuse the empirical-CDF
definition from the September landmark atlas and the August COPD TRE notebook's
post-step presentation, using current audited native snap TRE only. Plot Initial
and all three methods as percentage with error<=tolerance. Cache both/left/right
series before Ready, retaining inverse-location exceptions. A TRE-order sweep
marks the selected error and the upper edge of its CDF jump, including all ties;
anatomical/random order has no sweep marker. Keep method identity/colors stable
through column reorder, and synchronize both navigation bars and cloud picking.

Default to full range; offer0–6mm and a positive editable maximum. Navigation
never changes the axes; explicit zoom survives side filtering and a new case
resets full range. Show off-scale cursor status. Use local SVG and existing
state, no additional dependency, requests, scientific outputs or volume rebuild.
Validate themes, tie/zero counts, filtered denominators, navigation, disposal,
unchanged case receipts, four-case hardware response and request-free Ready.
Keep evidence under external validation/ui_r4_tre_cdf_port1101/.

Change the serving default/examples to loopback1101. Recreate the private-LAN
and loopback services with the existing pinned image/read-only mounts on1101;
validate replacements before stopping8765 services, retaining old containers.
Host/interface details remain private. No staging, commit or publication.

## User UI revision R5: TRE beneath landmark column titles

Remove both standalone TRE strips from the landmark tab. Put Initial TRE on
its own line immediately below the Moving MPR title and residual TRE below
each method title, retaining two decimal places. Fixed has no TRE subtitle.
Preserve both navigation bars and the CDF readout. Bind header values by method
identity so navigation, lung filtering, point picking and column reorder stay
synchronized. Keep graph-guard provenance below the TRE subtitle. No image-tab
header or scientific-data change. Validate normal/guarded headers, both themes,
both navigation bars and request-free interaction; retain new evidence under
external `validation/ui_r5_header_tre/`.
