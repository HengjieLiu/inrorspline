# DIR-Lab private case viewer

## Separate three-grid comparison

Use the same accepted `BUNDLE_ROOT` with a separate service:

```bash
python3 -B -m scripts.visualization.lungct_viewer.serve \
  --bundle-root "$BUNDLE_ROOT" --viewer grid-compare --host 127.0.0.1 --port 1102
```

This page has CT, differences from fixed, analytic pull grids, raster pull
grids and projected mapped meshes, with an empty second panel. Grid spacing
defaults to8 display voxels. Apply grids calculates/caches every coronal slice
before enabling navigation; subsequent scrolling requires no calculations or
requests. Rows3/4 are pull patterns in fixed space; row5 maps fixed mesh vertices
into moving space and projects onto L/S, without contours. All three hide AP
motion. The existing standard page/default port are unchanged by this option.
See `docs/implementation/lungct_grid_compare.md` for geometry and verification.

One accepted case in browser memory, two tabs, shared method order. The first
tab shows coronal CT, differences from fixed, pull grids, magnitude and signed
SI. The second shows moving/fixed3D landmark clouds, four TRE cumulative
curves, and3x5 orthogonal CT views.
All interaction after Ready uses already loaded data. No registration runs.

## Prepare

Set `BUNDLE_ROOT` outside Git, normally
`${ARTIFACT_ROOT}/prepared/lungct_viewer/<unique-version>`. The accepted P87
campaign supplies `CAMPAIGN_REVIEW` and `PREPARED_DATA`; `MASK_ROOT` names the
native R231 left/right labels. These are command arguments, not fixed paths.
Prepare inside the established Docker runtime, with source/raw mounts read-only,
artifact storage writable, and the existing h5py/SciPy dependency overlay.

```bash
bash scripts/visualization/lungct_viewer/fetch_assets.sh "$BUNDLE_ROOT"
python -B -m scripts.visualization.lungct_viewer.prepare \
  --campaign-review "$CAMPAIGN_REVIEW" --data-root "$PREPARED_DATA" \
  --mask-root "$MASK_ROOT" --output-root "$BUNDLE_ROOT"
```

`--cases 4dct:01` runs the small real-case smoke. `--resume` verifies/reuses
accepted cases and completes the remaining cohort; existing unaccepted case
directories block overwrite. Retain failures and use a fresh case destination
after correction. Catalog entries appear only after the case receipt passes.
Three.js0.180.0 is downloaded once, checked against fixed SHA256 pins, and served
locally. The browser makes no CDN requests. Node is not required to serve the app.

## Open

```bash
python3 -B -m scripts.visualization.lungct_viewer.serve \
  --bundle-root "$BUNDLE_ROOT" --host 127.0.0.1 --port 1101
```

The server uses only Python's standard library. Run it on the data host and
forward port1101 through VS Code or SSH:

```bash
ssh -N -L 1101:127.0.0.1:1101 YOUR_SERVER
```

Open **http://127.0.0.1:1101** in a desktop browser with WebGL2 and hardware
acceleration. Keep the server running while browsing. It serves only static
source, the catalog, approved case assets and pinned dependencies. It has no
write or job-launch API; private provenance paths are removed from HTTP manifests.
For private-network access, bind explicitly with `--host YOUR_PRIVATE_INTERFACE`
and open `http://YOUR_PRIVATE_INTERFACE:1101/`; no tunnel is needed in that mode.

Choose4DCT/COPD and a case. Loading includes download/decode/GPU upload and both
tabs' shader initialization. The previous case is released on selection. Ready
means scrolling, ordering, filtering and tab changes need no further data request.
Earlier/Later method controls update both tabs and persist in browser storage.
The header's Light mode toggle switches the page and point-cloud background;
the preference persists across reloads. CT windowing and scientific map colors
retain the same meaning in either theme.
Scroll over comparison images to change the shared coronal plane; scroll the
page outside an image to reach lower rows. Landmark Previous/Next wraps; keyboard
left/right works outside input controls. Cloud points can be selected by clicking.
Previous/Next, the landmark slider and position/ID are also available directly
below the orthogonal images. The upper and lower controls share one selection
and update together, including filtered lists and wraparound. Initial TRE is
shown on a new line below the Moving column title; each method's residual TRE
is below its own title. These two-decimal subtitles update with selection and
follow method identity when columns are reordered. Fixed has no TRE subtitle.

The TRE plot shows Initial and the three methods as percentage with audited
native snap error<=tolerance. Both/left/right lung selection changes the curves
and denominator together; inverse-location exceptions remain included. Full
range is the default;0–6mm and an editable maximum zoom the tolerance axis.
Explicit zoom survives lung filtering; each new case starts at full range.
During TRE ordering, a vertical line marks the selected landmark's error for
the ranking metric, highlighting that curve's tie-inclusive cumulative count.
Tied points can share an intersection despite different navigation positions.
Anatomical/random ordering hides this sweep marker. Off-scale markers are
explicit, colors follow method identity through reordering, and displayed
two-decimal values do not round the curve inputs. SVG interactions use only
the already loaded records; no additional plot dependency or data request.

## Geometry and interpretation

- Physical coordinates are pair-local LPS, not reconstructed scanner-world DICOM.
  Raw X increases toward patientL, raw Y towardP, raw Z towardI. Coronal S is up
  and patientR is screen left; axial A is up; sagittal A is screen left.
- Every field retains its original crop and interpolation geometry. The common
  display grid's spacing comes from native endpoints; it is not blindly1mm.
- Raw-HU images are warped in3D using each original pull field, including
  through-plane motion. Source nominal1mm displacement units are millimetres;
  they are distinct from endpoint-derived display-grid spacing.
- Cyan contours follow moving/warped anatomy. Yellow is the fixed target.
  The difference row is moving-minus-fixed, each deformed-minus-fixed, and
  fixed-minus-fixed (zero), in HU before
  windowing. Default difference range is at most±1000HU; larger values saturate.
- Total magnitude and signedSI share one symmetric color scale. Magnitude uses
  its nonnegative half. SignedSI is the pull component, positive toward superior;
  it is not a forward anatomical trajectory map. Grid spacing defaults10mm.
- Landmark centers solve the inverse pull equation without fixed labels as
  seeds. Inversion residual≤0.05mm is required. Multiple roots/unresolved points
  remain explicit; unresolved columns center on the fixed target with no invented
  moving crosshair. Projected target markers state their distance from the plane.
- In the slice views, cyan crosshairs mark the selected moving/transformed
  point; the yellow ring is a separate fixed target projected into that plane.
  They coincide in the Fixed column. The target marker has an independent toggle
  from the fixed contour and selected-point crosshairs. TRE labels use two decimals;
  stored values and ordering retain their original precision.
- Full crop keeps complete in-plane bounds and slices through the chosen point.
  The10/20/50mm choices center their square in-plane extent on that point. Padding
  preserves millimetres near boundaries; offscreen fixed targets are noted.
- TRE values/sorting are unchanged audited native snap values. Inverse markers
  use continuous coordinates and do not redefine TRE. MR/pTV and CorrField have
  different optimization intensity/mask protocols, despite common display HU.
  Guarded CorrField4DCT04–06 have visible method tags.

## Verify

```bash
python -B -m pytest -q -p no:cacheprovider tests/test_lungct_viewer.py
python -B -m scripts.visualization.lungct_viewer.validate \
  --bundle-root "$BUNDLE_ROOT" --campaign-review "$CAMPAIGN_REVIEW" \
  --data-root "$PREPARED_DATA" --mask-root "$MASK_ROOT" \
  --output "$BUNDLE_ROOT/validation/new_data_audit.json"
```

The independent audit reloads every asset, checks original source hashes and
landmark text, and reconstructs sampled HU warps, vectors, masks and inverse
locations directly from original fields. It requires all20 cases unless the
explicit `--allow-incomplete` smoke switch is used. Validation outputs are new
files and never replace earlier reports.

`browser_check.mjs` uses a separately installed Playwright1.56.1. Set
`VIEWER_PLAYWRIGHT_MODULE` to its `index.mjs`, `PLAYWRIGHT_BROWSERS_PATH` to the
external browser install, and `VIEWER_VALIDATION_ROOT` to a new external directory.
Run with Node22.16.0. `VIEWER_ANGLE=gl-egl` requests hardware rendering; default
SwiftShader measures software rendering and cannot certify desktop performance.
The report records the actual renderer, four representative cases, request-free
interaction, performance and case cancellation/disposal. Screenshots remain external.
It includes the TRE chart's tie-inclusive cursor, all side/metric/direction
combinations, range persistence and sustained landmark-navigation timing.
`node --test tests/lungct_viewer_tre_cdf.test.mjs` checks the zero/tie/precision
conventions without browser dependencies. For UI-only revisions, retain the
original acceptance receipt and seal new evidence in a separate validation
revision directory; do not rerun finalization over an accepted bundle.

After the complete data audit and four-case hardware browser check pass, run
`python -B -m scripts.visualization.lungct_viewer.finalize --bundle-root "$BUNDLE_ROOT"`.
It verifies both gates and unchanged sources, then creates an external validation
report, inverse-exception CSV, source snapshot and final acceptance receipt.
