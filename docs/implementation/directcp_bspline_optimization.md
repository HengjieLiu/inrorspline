---
doc_type: implementation_summary
status: active_draft
---

# DirectCP And Cubic B-spline Optimization

DirectCP is the direct control-point baseline. It removes the INR/SIREN control
generator and optimizes the cubic B-spline control displacements themselves.

## DirectCP Parameterization

For a control grid with shape `[Zc, Yc, Xc]`, DirectCP stores:

```text
delta[z, y, x, 3]
```

as trainable parameters. Each control value is a normalized-grid displacement
component. The dense displacement field is obtained by cubic B-spline expansion:

```text
u = expand_cubic_bspline(delta)
phi(p) = p + u(p)
```

DirectCP uses Adam. In OASIS runs, single-stage DirectCP cps4 and cps2 optimize
global NCC plus analytic cubic B-spline BE. In LungCT pTV-prep runs,
single-stage DirectCP cps4 and multistage DirectCP optimize pTV-style local CC
plus isoTV, while diagnostic variants optimize global NCC plus either analytic
B-spline BE or dense forward-difference BE.

## Cubic B-spline Basis

The scalar cubic B-spline basis used by the code has compact support on
`[-2, 2]`. Let `t = |x|`.

```text
B(x) =
  2/3 + 0.5*t^3 - t^2      if 0 <= t < 1
  (2 - t)^3 / 6            if 1 <= t < 2
  0                        if t >= 2
```

The first derivative is:

```text
B'(x) =
  (1.5*t - 2) * x          if 0 <= t < 1
  0.5*(t - 2)^2            if 1 <= t < 2 and x < 0
 -0.5*(t - 2)^2            if 1 <= t < 2 and x >= 0
  0                        if t >= 2
```

The second derivative is:

```text
B''(x) =
  3*t - 2                  if 0 <= t < 1
 -t + 2                    if 1 <= t < 2
  0                        if t >= 2
```

For a 3D control lattice `c[i,j,k]`, the dense displacement at voxel-center
position `p = (z, y, x)` is separable:

```text
u(p) = sum_i sum_j sum_k c[i,j,k]
       * B((z - z_i) / cps)
       * B((y - y_j) / cps)
       * B((x - x_k) / cps)
```

The implementation realizes this with separable transposed 1D convolutions and
then crops the expanded field to the image shape.

## Analytic B-spline Derivatives

Analytic derivative fields are computed by replacing the basis kernel along an
axis with the derivative kernel for that axis. A derivative of order `q` along
an axis is scaled by:

```text
(1 / cps)^q
```

because the basis is evaluated as `B((x - x_i) / cps)`.

Diffusion energy uses first derivatives:

```text
diffusion(u) = mean over components and voxels of
  (du/dz)^2 + (du/dy)^2 + (du/dx)^2
```

Bending energy uses second and mixed derivatives:

```text
BE(u) = mean over components and voxels of
  u_zz^2 + u_yy^2 + u_xx^2
  + 2*u_zy^2 + 2*u_zx^2 + 2*u_yx^2
```

In the OASIS runner these analytic quantities are sampled at voxel centers and
reported in both normalized-grid and voxel units where possible. In the LungCT
runner, analytic B-spline BE/diffusion are used for diagnostic/global-NCC BE
variants and posthoc summaries where available.

Analytic B-spline BE/diffusion are the native regularity metrics for spline
methods because they measure derivatives of the continuous parameterized field,
not only derivatives of a sampled dense array.

## Finite-Difference Metrics

Dense finite-difference metrics operate on the sampled dense displacement
field. They are useful for common reporting across methods and saved baseline
dense fields.

Forward-valid first differences are used for dense diffusion:

```text
dz = u[z+1,y,x] - u[z,y,x]
dy = u[z,y+1,x] - u[z,y,x]
dx = u[z,y,x+1] - u[z,y,x]
```

Forward-valid second and mixed differences are used for dense BE:

```text
dzz, dyy, dxx, dzy, dzx, dyx
```

with the same BE weights as the analytic formula. Valid masks require all
voxels touched by a finite-difference stencil to be inside the foreground.

Central differences are used in separate Jacobian/log-Jacobian computations.
Those metrics estimate local deformation gradients from the dense sampled
field, crop away boundary voxels, and are not the same quantity as analytic
B-spline BE.

Practical interpretation:

- Analytic B-spline BE/diffusion: native for SINR and DirectCP spline fields.
- IDIR autograd BE/diffusion: native for dense SIREN IDIR.
- Dense forward differences: common dense-field audit metric for all methods,
  including saved baselines.
- Central differences: Jacobian/log-Jacobian style diagnostics.

## Multistage DirectCP

Multistage DirectCP changes the optimization path while preserving a final
cubic B-spline representation. It starts with a coarse control spacing and
progressively inserts knots to refine the control lattice.

Current schedules:

- OASIS preliminary multistage: full cps `8,4,2`, downsample `4,2,1`.
- LungCT main multistage: full cps `32,16,8,4`, downsample `8,4,2,1`.

The downsample factor defines the image domain used for a stage. The effective
stage-domain control spacing is:

```text
domain_cps = full_cps / downsample
```

The code requires `full_cps` to be divisible by `downsample`.

## Nested Grid Criteria

The implementation uses `CP_MARGIN = 4` and constructs nested control shapes
from a base spacing. Consecutive stages must be dyadic:

```text
coarse_cps = 2 * fine_cps
```

The fine control-grid shape must satisfy:

```text
fine_shape = 2 * (coarse_shape - 4) + 4
```

This condition makes the crop-aligned cubic knot insertion exact for the
supported schedules. If the condition fails, the runner raises an error instead
of silently resizing controls.

## Knot Insertion

For each axis, crop-aligned dyadic cubic knot insertion maps a coarse sequence
`c_i` to a finer sequence by alternating two coefficient formulas:

```text
even = 0.5 * (c_i + c_{i+1})
odd  = (c_i + 6*c_{i+1} + c_{i+2}) / 8
```

The operation is applied along z, y, and x. The result initializes the next
stage's finer control lattice. Preservation diagnostics expand the inserted
finer controls back to a dense field and compare it against the previous
stage's dense field.

Trilinear resizing is not the canonical prolongation because it resizes control
values as an image-like array. Exact cubic knot insertion preserves the
continuous B-spline field under the nested-grid assumptions, which is the
behavior needed for a fair coarse-to-fine optimization path.
