# DIR-Lab three-grid comparison

Approved 2026-11-19: a separate read-only viewer on port1102, all20 accepted
P88 bundles, one case resident, five coronal columns and five rows. The existing
standard viewer is preserved (currently1101 after its separately recorded R4
update; historical8765 containers remain stopped and retained).

## Geometry and display

Rows: raw-HU CT; moving/deformed minus fixed and fixed-minus-fixed; analytic
pull grid; raster pull grid; projected mapped fixed mesh. Moving/fixed cells
are blank in grid rows. Shared pair-local LPS, S up/R left, endpoint-derived
physical spacing, preserved crop/field validity and guarded CorrField tags.
Methods default MR-D-BSCP, pTVReg, CorrField with reorder controls. Panel2 is empty.

All constructions start at display index0 and use integer display-voxel spacing,
initially8. Convert physical uL/uS using signed display steps; never rescale the
saved displacement itself. Analytic lines are level sets of source indices.
Raster lines occupy integer indices divisible by spacing, one voxel wide,
extruded through AP, linearly sampled with border clamping. Mapped mesh samples
fixed grid lines at <=1voxel intervals, maps with plus displacement, then projects
onto L/S; invalid segments omitted, output clipped. Rows3/4 are fixed-space pull
patterns; row5 is a moving-space projection with no contours. AP motion is hidden
in all three coronal patterns. Contour toggles apply only to rows1--4.

## Cache and service

Build nine single-channel R8 GPU array textures per case/spacing, one layer per
coronal slice, before Ready. Apply spacing explicitly; disable scrolling while
building a replacement, keep last completed view, then swap atomically. Bound
memory to active plus one pending cache; cancel/dispose obsolete jobs. Batch GPU
work with UI yields. After Ready interactions use cached textures only, with no
requests or reconstruction. Record cache seconds/bytes and hardware performance.

Source in scripts/visualization/lungct_viewer; separate app and tests. CLI adds
--viewer standard|grid-compare with unchanged standard default/port. New CPU-only
Docker service uses pinned existing image, RO repo/artifact mounts, private host
bind1102. Existing services and scientific assets unchanged. Generated evidence
under ${ARTIFACT_ROOT}/prepared/lungct_grid_compare/v1_20261119/. No registration,
raw-data preparation, staging, commit, push, publication or source synchronization.

## Acceptance

Zero, signed translation, affine scaling, nonlinear curves, raster sampling,
validity/clipping, contour/orientation and difference tests. Audit20 manifests
and260 asset hashes. Hardware browser checks 4DCT01/02,COPD01/10: all rows,
spacing8/other/back, busy gating, cancellation/switching, order/theme/contours,
empty panel2, >=30fps/p95<100ms after Ready, no new requests/memory accumulation.
Regression check both standard tabs on its current1101 service. Preserve failed
attempt receipts. Record limitation: display DVFs have audited <=0.1mm compact
error; projected mesh is not an inverse field or anatomical forward trajectory.
