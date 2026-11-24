# Three coronal grid constructions

The separate comparison application uses the accepted P88 case bundles without
preparing new CT, masks, fields or landmarks. Start the read-only server with
`--viewer grid-compare --port 1102`; the standard page selector remains the
default. Runtime host bindings are private operational configuration.

## Coordinate contract

Display indices use the saved endpoint-aligned physical geometry. At fixed
index `(x,y,z)`, the moving sampling indices in the coronal plane are
`(x+uL/stepL, z+uS/stepS)`, where `stepS` is negative. Physical vectors are not
reinterpreted as display-voxel vectors without this conversion. The source
grid phase is index0, with integer spacing initially8 display voxels; the
interface reports the two physical spacings separately.

Analytic pull lines are coordinate level sets with derivative antialiasing.
The raster pull pattern has one-voxel binary lines at indices divisible by
spacing, repeated along AP, sampled bilinearly with edge replication. This is
equivalent to trilinear sampling that AP-constant3D grid with border padding.
Analytic coordinates repeat outside the source FOV; raster padding can extend
edge lines. They intentionally retain this difference in boundary behavior.

The mapped mesh starts with regular fixed-space lines, segmented at<=1voxel.
Vertices use the same plus-displacement map and are projected onto moving L/S
coordinates. Segments with invalid endpoints are omitted; the output is clipped
to the shared physical crop. Output positions are not masked by fixed-space
validity, which would incorrectly erase projected segments. Row5 receives no
fixed-space contours. All three patterns hide AP motion; CT images retain the
full accepted3D warp. This mesh is not an inverse map or a moving-to-fixed
material trajectory, and projected crossings alone do not establish3D folding.

## Cache lifecycle

Nine WebGL2 R8 array textures hold line coverage at display-grid resolution:
three methods by three constructions, one coronal layer per texture. GPU bake
passes yield between batches. A completion fence and shader warmup precede
Ready. The raster grid's additional coverage quantization is at most1/255;
scientific FP32 outputs and their existing compact-display bounds are unchanged.

Apply builds a replacement while keeping the previous completed view. Slice
controls, wheel and keyboard navigation are gated during the calculation.
Revision tokens cancel obsolete builds; serialization bounds allocations to one
active plus one pending cache. Successful replacement disposes the previous
cache. Case changes abort downloads, invalidate builders and dispose textures,
materials and CPU array references. New cases use the last applied spacing,
discarding uncommitted spacing text. No post-Ready data requests or grid
reconstruction are required for scrolling, ordering, contours or themes.

## Evidence

Specification/log: `experiments/2026-11-19_lungct_grid_compare/`. Generated
receipts, failed attempts, screenshots and browser measurements are external
under `${ARTIFACT_ROOT}/prepared/lungct_grid_compare/v1_20261119/validation/`.
Synthetic GPU tests cover identity, translation, scaling, nonlinear fields,
invalid regions, raster equivalence and cancellation. Four real cases cover
spacing replacement, cancellation/switching, contours, themes, request-free
interaction, memory and hardware performance; the main viewer is checked
separately. Scientific-data integrity uses all20 manifest receipts/260 hashes.
