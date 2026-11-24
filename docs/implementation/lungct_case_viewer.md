# DIR-Lab case viewer

The viewer is a posthoc consumer of P87 accepted outputs. It never launches
registration or modifies the scientific inputs. Source lives in
`scripts/visualization/lungct_viewer/`; generated bundles, provenance, dependency
assets and validation results live under
`${ARTIFACT_ROOT}/prepared/lungct_viewer/<version>/`.

## Geometry

The physical frame is pair-local anatomical LPS, with raw XYZ mapped to L/P/I.
For raw native index `(x,y,z)`, use
`(x*sx, y*sy, (native_depth-1-z)*sz)`. This makes superior coordinates increase
upwards and patient right appear screen left in coronal views. It is not a
claim about recovered DICOM scanner origins. Axial shows A at top and R at left;
sagittal shows S at top and A at left.

Grid endpoints, crop origins and displacement units are separate. The display
grid spacing is `(native_crop_shape-1)*native_spacing/(registration_shape-1)`.
Original MR/CorrField vectors are nominal1mm voxel displacements, interpreted
as millimetres exactly as in the accepted native evaluator. They must not be
multiplied by display-grid effective spacing. Historical pTV fields are native
XYZ millimetres and retain their archived one-based crop bounds.

The complete original3D pull field samples raw moving HU at each physical
display-grid position. R231 labels use nearest neighbor through the same map.
The exported lightweight map stores `(uL,uS,magnitude,valid)` for shader grids
and deformation colors; the full original field remains unchanged externally.
Half-float display conversion requires≤2HU/0.1mm error; otherwise it usesFP32.

## Landmarks

Native snap errors and ordering values are copied from the audited CSV and
replayed with the original reduction order: ZYX for MR/CorrField, XYZ for pTV.
Even FP64 norm sums can differ in the last bit when component order changes.
The independently computed initial distance uses native physical coordinates.

To locate a moving landmark in a deformed image, solve `q+u(q)=moving_point`.
Coarse mapped-grid nearest-neighbor seeds and an identity seed initialize
multistart damped Newton, without using the fixed counterpart as a seed.
Solutions require residual≤0.05mm; distinct roots are tagged ambiguous.
Unresolved inversions retain their correspondence and show a labelled fixed
target center without a fictitious transformed crosshair.

Native snap TRE is not recomputed from inverse markers. MPR images are centered
on continuous transformed positions, with projected fixed-target markers and
out-of-plane distance annotations. Full view preserves complete in-plane crop
bounds; small ROIs have physical10/20/50mm side lengths and explicit padding.

## Browser pipeline

The app is buildless ES modules with pinned, locally served Three.js0.180.0.
A shared WebGL2 renderer reuses3D textures across both tabs. Preparation exports
HU volumes, nearest-neighbor label volumes and compact map volumes once. Case
selection downloads/decodes/uploads all assets and warms both tabs before Ready.
An anchor material retains the compiled shader during method reordering.
Interactions update slice coordinates, uniforms or small point buffers only.
Case changes abort obsolete loads and dispose old textures, geometries, controls
and render lists. Inactive/offscreen views are not drawn.

The standard-library HTTP server defaults to loopback1101, supports GET/HEAD
and private caching, and has no write API. Only approved case assets and pinned
dependencies are exposed; private source paths are stripped from case manifests.
The service does not need CUDA, NumPy, Node, a build server or a runtime CDN.

## Interpretation and verification

The image tab uses raw-HU display for all methods, whereas their optimization
protocols still differ. CorrField forest-guard4DCT04–06 tags remain visible.
Total magnitude and signed superior pull displacement share a symmetric color
range; magnitude occupies only the nonnegative half. Following the user's UI
revision, every difference cell subtracts the fixed CT: moving-minus-fixed,
three deformed-minus-fixed cells and the fixed-minus-fixed zero reference.
Differences are evaluated before windowing on the already loaded HU textures.

The two-decimal TRE readouts do not round stored values or sort keys. A visible
legend distinguishes selected-point crosshairs from projected fixed-target rings;
their toggles are independent of fixed-mask contours. Light/dark selection is
saved in browser storage and changes page/point-cloud presentation and padding,
while CT intensities, signed map polarity and physical geometry remain unchanged.
Landmark navigation is duplicated directly below the MPR grid. Both slider
ranges/values and labels derive from the same filtered list/index; both button
sets call the same selection handler. R5 removes the two standalone TRE strips
and places Initial TRE directly below the Moving MPR title and residual TRE
below each method title. Header elements are cached by method key and updated
from the same selected correspondence, retaining two decimals through column
reorder. Fixed has no TRE subtitle; graph-guard tags follow the TRE line.

The third panel beside Moving/Fixed clouds renders four SVG empirical TRE
step distributions. The historical atlas/notebook definition is reused as
`100*count(error<=tolerance)/N`, with current native snap inputs, no smoothing
or display rounding. Both/left/right series are cached before Ready; inverse
exceptions retain their benchmark errors and contribute to the denominator.
Paths change only with case/side/range; navigation updates the selected-metric
vertical line, tie-inclusive intersection and readout. Anatomical/random order
hides the sweep marker. Method identity survives column reorder; theme colors
retain each method's hue. Full range covers the filtered four-method maximum;
manual/0–6mm ranges persist through side filtering and reset on a new case.
Out-of-range markers are labelled rather than changing axes during navigation.
The chart adds no dependency, volume processing or post-Ready request.

Synthetic tests cover geometry, field sign/axes, masks, differences, inverse
ambiguity/failure and read-only HTTP behavior. The external data audit independently
reconstructs sampled images/maps/masks and inverse positions from original fields,
checks original landmark text and all asset/source hashes. Browser checks record
the actual graphics backend; software-rendered timing is kept separate from the
hardware60fps target/30fps and p95response100ms acceptance gate.
