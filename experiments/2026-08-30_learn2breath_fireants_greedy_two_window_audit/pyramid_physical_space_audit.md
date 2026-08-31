# FireANTs Greedy NCC vs MR-D-BSCP P2e: physical pyramid audit

## Scope and conclusion

This is a source-and-run-metadata audit of the two **already completed**
CT-only registrations on the Learn2Breath-style grid.  It does not rerun or
reinterpret their TRE/Dice results.

The nominal schedules both read `16 -> 8 -> 4 -> 2 -> 1`, but they do not
describe the same physical optimization path.  In the pinned FireANTs build,
`MIN_IMG_SIZE = 32` clamps both requested `16` and requested `8` levels to the
same `32 x 32 x 32` image.  Consequently, FireANTs never visits the genuine
`16x` or `8x` global image levels used by P2e.  Its first two 200-iteration
stages are two optimizer passes on the same approximately 10 mm lattice.

Thus, the concern is supported only in this precise sense: FireANTs lacks the
two *coarsest global-capture image stages*.  It is not using a coarser early
lattice than P2e: its earliest actual image and dense deformation lattices are
much **finer** and have far more degrees of freedom.  The two methods also
differ in deformation representation, local-similarity support, smoothing,
regularization, and iteration allocation, so this audit identifies a strong
schedule/scale mismatch rather than proving a single cause of the TRE gap.

## Shared physical reference frame

Both completed audits use the same full fixed grid:

- Shape: `X x Y x Z = 224 x 192 x 224` voxels.
- Spacing: `1.5 x 1.5 x 1.5 mm`.
- Voxel-centre extent: `334.5 x 286.5 x 334.5 mm`, calculated as
  `(shape - 1) * spacing`.
- Resampling convention for the audit calculations: endpoint-preserving
  `align_corners=True`.  Therefore the spacing of a resampled axis is its
  full-grid physical extent divided by `(resampled_length - 1)`, not simply
  `1.5 mm * requested_scale`.

Shapes below are written in `X x Y x Z` order.  The implementation stores
PyTorch tensors in `Z, Y, X` order; X and Z happen to be equal for this grid.

## What the nominal image schedule becomes

| Requested level | FireANTs actual shape | FireANTs actual voxel spacing (mm, X x Y x Z) | P2e actual shape | P2e actual voxel spacing (mm, X x Y x Z) | FireANTs iterations | P2e iterations |
|---:|---|---|---|---|---:|---:|
| 16 | `32 x 32 x 32` | `10.79 x 9.24 x 10.79` | `14 x 12 x 14` | `25.73 x 26.05 x 25.73` | 200 | 400 |
| 8 | `32 x 32 x 32` | `10.79 x 9.24 x 10.79` | `28 x 24 x 28` | `12.39 x 12.46 x 12.39` | 200 | 400 |
| 4 | `56 x 48 x 56` | `6.08 x 6.10 x 6.08` | `56 x 48 x 56` | `6.08 x 6.10 x 6.08` | 200 | 400 |
| 2 | `112 x 96 x 112` | `3.01 x 3.02 x 3.01` | `112 x 96 x 112` | `3.01 x 3.02 x 3.01` | 200 | 400 |
| 1 | `224 x 192 x 224` | `1.50 x 1.50 x 1.50` | `224 x 192 x 224` | `1.50 x 1.50 x 1.50` | 200 | 900 |

Two implications follow directly.

1. P2e starts with image samples approximately `26 mm` apart, then `12.4 mm`,
   before reaching the common `6.1`, `3.0`, and `1.5 mm` levels.  This is a
   true coarse-to-fine sequence across five distinct image lattices.
2. FireANTs starts at approximately `10.8 x 9.2 x 10.8 mm`, repeats that exact
   lattice for the nominal 16 and 8 stages, then shares the final three image
   lattices with P2e.  Its requested 16x level is effectively only
   `7.19 x 6.16 x 7.19` times the native voxel interval, and the requested 8x
   level is identical—not an 8x level.

FireANTs has `1,000` optimizer iterations in total; P2e has `2,500`.  The
first FireANTs 400 iterations do not introduce a new image scale between them;
the stage transition resets the per-level optimizer/convergence machinery on
the same 32-cubed lattice.  P2e instead allocates 800 iterations to two
different, substantially coarser physical scales and 900 at full resolution.

## Deformation representation: physical control scale versus dense lattice

Image resolution alone is not the deformation prior.  P2e optimizes a cubic
B-spline control grid and transfers it by exact dyadic knot insertion.  Its
declared control spacings are full-grid voxel units `64 -> 32 -> 16 -> 8 -> 4`,
which correspond nominally to `96 -> 48 -> 24 -> 12 -> 6 mm` on this data.

Because each P2e image level represents four control-grid intervals and uses
endpoint-preserving resampling, the exact interval in the current level is
`4 * level spacing`; it is slightly different from the nominal value at coarse
levels.  The following table reports that exact interval and the number of
optimized vector coefficients.

| P2e stage | Image shape | B-spline control shape (X x Y x Z) | Exact control interval (mm, X x Y x Z) | Vector coefficients |
|---:|---|---|---|---:|
| `cp64`, image 16 | `14 x 12 x 14` | `8 x 7 x 8` | `102.92 x 104.18 x 102.92` | 1,344 |
| `cp32`, image 8 | `28 x 24 x 28` | `12 x 10 x 12` | `49.56 x 49.83 x 49.56` | 4,320 |
| `cp16`, image 4 | `56 x 48 x 56` | `20 x 16 x 20` | `24.33 x 24.38 x 24.33` | 19,200 |
| `cp8`, image 2 | `112 x 96 x 112` | `36 x 28 x 36` | `12.05 x 12.06 x 12.05` | 108,864 |
| `cp4`, image 1 | `224 x 192 x 224` | `68 x 52 x 68` | `6.00 x 6.00 x 6.00` | 721,344 |

By contrast, the FireANTs Greedy configuration uses a dense compositive warp:
one 3-vector per voxel of its current registration lattice.  It does not use
a B-spline control grid or P2e-style knot insertion.  Its learned lattice size
and raw vector-parameter count are therefore:

| FireANTs requested stage | Actual dense warp lattice | Dense vector parameters | Approximate smoothing sigma in physical mm, warp / gradient |
|---:|---|---:|---|
| 16 | `32 x 32 x 32` | 98,304 | `2.70 x 2.31 x 2.70` / `5.40 x 4.62 x 5.40` |
| 8 | `32 x 32 x 32` | 98,304 | `2.70 x 2.31 x 2.70` / `5.40 x 4.62 x 5.40` |
| 4 | `56 x 48 x 56` | 451,584 | `1.52 x 1.52 x 1.52` / `3.04 x 3.05 x 3.04` |
| 2 | `112 x 96 x 112` | 3,612,672 | `0.75 x 0.75 x 0.75` / `1.51 x 1.51 x 1.51` |
| 1 | `224 x 192 x 224` | 28,901,376 | `0.38 x 0.38 x 0.38` / `0.75 x 0.75 x 0.75` |

The smoothing conversion is deliberately labelled approximate: it multiplies
the configured `smooth_warp_sigma=0.25` and `smooth_grad_sigma=0.5` pixels by
the endpoint-preserving level spacing.  It conveys the physically relevant
fact that FireANTs' smoothing becomes progressively narrower in millimetres as
resolution increases.  This run passes neither `displacement_reg` nor
`warp_reg`; its configured smoothers are not an explicit P2e/pTV-style
physical isoTV penalty.

At the nominal 16 stage FireANTs consequently has 98,304 dense vector
parameters at a roughly 10 mm lattice, whereas P2e has 1,344 B-spline vector
coefficients with a roughly 103 mm control interval.  Even where the image
lattices match (requested 4, 2, and 1), FireANTs optimizes substantially more
local degrees of freedom and P2e maintains a coarser, structured deformation
parameterization.

## Similarity support in physical space

The methods do not merely use different forms of NCC; their locality changes
in different physical ways over the pyramid.

### FireANTs Greedy NCC

The fixed configuration is rectangular local NCC with a 5-voxel kernel at
every actual image level.  The unambiguous distance from the first to the last
of the five kernel sample centres is four lattice intervals:

| Actual image lattice | 5-sample outer-centre span (mm, X x Y x Z) |
|---|---|
| `32 x 32 x 32` | `43.16 x 36.97 x 43.16` |
| `56 x 48 x 56` | `24.33 x 24.38 x 24.33` |
| `112 x 96 x 112` | `12.05 x 12.06 x 12.05` |
| `224 x 192 x 224` | `6.00 x 6.00 x 6.00` |

This is a window-centre span rather than a claim about the exact boundary of
the discrete NCC implementation.  It shows that the physical neighbourhood is
large at the repeated 32-cubed stage and contracts sharply to 6 mm at full
resolution.

### P2e pTV-style local CC

P2e uses a pTV-style local CC target sigma of `2.1 mm`, with radius-sigma value
`2.0`, plus a minimum sigma of `0.8` *level pixels*.  Therefore the minimum
pixel rule dominates on the first four P2e levels.  The effective Gaussian
sigma is:

| P2e image level | Effective LCC sigma (mm, X x Y x Z) |
|---|---|
| `14 x 12 x 14` | `20.58 x 20.84 x 20.58` |
| `28 x 24 x 28` | `9.91 x 9.97 x 9.91` |
| `56 x 48 x 56` | `4.87 x 4.88 x 4.87` |
| `112 x 96 x 112` | `2.41 x 2.41 x 2.41` |
| full | `2.10 x 2.10 x 2.10` |

Thus P2e's similarity locality also changes at coarse resolution, but it is
defined through a physical-mm target and becomes exactly 2.1 mm at the final
grid.  FireANTs' fixed 5-pixel kernel instead maps to a 6 mm outer-centre span
at full resolution and much larger spans on its coarser lattices.

## Interpretation for the observed COPD motion deficit

The fixed-lung audit for case 104 found a smaller FireANTs displacement
magnitude distribution than P2e, while the independently checked FireANTs
TRE evaluator, field sign/axis convention, and NIfTI reload all agreed with
the saved outputs.  This physical-pyramid audit adds a credible optimization
mechanism:

- P2e can first explain respiratory motion with fields whose initial degrees
  of freedom vary at approximately 100 mm and then 50 mm scales, before
  admitting 24, 12, and 6 mm detail.
- FireANTs begins with a 10 mm dense warp, smoothed at only about 5 mm in its
  gradient path, and never presents the optimizer with P2e's very coarse
  image/control scales.  Its two nominally coarsest stages are duplicates.
- Starting finer does not mean greater capture range.  In a large,
  intensity-only breathing deformation it can instead make the early search
  more local and admit many non-global updates before broad correspondence is
  established.

This is a **plausible explanation, not a causal proof**.  The old audit changes
several factors at once: local similarity construction, optimizer and learning
rate, number of iterations, dense-compositive versus cubic-B-spline field,
smoothing/regularization, and transfer rule.  It would be inappropriate to
attribute the TRE difference solely to `MIN_IMG_SIZE` without a controlled
FireANTs ablation.

## Directly testable follow-up, if a new experiment is authorized

A controlled FireANTs schedule study would need a new protocol tag and must
not overwrite the completed audit.  It could test whether allowing genuine
`14 x 12 x 14` and `28 x 24 x 28` levels (by lowering the FireANTs minimum
image-size constraint) changes the result, while holding the intensity window,
field-export convention, metric, and all remaining FireANTs settings fixed.
Separately changing its iteration allocation to the P2e `400,400,400,400,900`
budget would answer a different question and should not be bundled with the
minimum-size ablation.

## Evidence inspected

- FireANTs run configuration:
  `scripts/experiments/learn2breath_fireants_greedy/run_pairlocal.py`
  (`SCALES`, `ITERATIONS`, 5-voxel rectangular NCC, compositive warp, and
  0.25/0.5-pixel smoothers).
- P2e run configuration and per-stage metadata:
  `scripts/experiments/learn2breath_mrdbscp_p2e/run_pairlocal.py`
  (`cp64 -> cp4`, image schedule, iteration schedule, and physical local-CC
  settings).
- Pinned FireANTs implementation used for the completed audit:
  `fireants.utils.globals.MIN_IMG_SIZE = 32` and the Greedy level-size rule
  `max(int(full_size / scale), MIN_IMG_SIZE)`.
- Completed P2e case-104 stage summaries, which confirm the actual image,
  control-grid, spacing, anti-alias, and epoch values reported above.
