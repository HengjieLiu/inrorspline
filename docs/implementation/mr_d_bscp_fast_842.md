---
doc_type: implementation_specification
status: active_draft
---

# MR-D-BSCP Fast 8-4-2: Implemented Method Specification

This document describes the repository implementation of the OASIS
`MR_A_fast_decay` candidate, reported as **MR-D-BSCP (8-4-2)**. It is intended
as a technical handoff for an independent acceleration audit. The description
follows the code rather than an idealized mathematical implementation; details
that are only true for other method families are explicitly excluded.

## 1. Method identity

MR-D-BSCP is a multiresolution direct control-point method:

- **MR**: the optimizer progresses through coarse, medium, and fine image
  domains;
- **D**: displacement control points are optimized directly;
- **BSCP**: the displacement is represented by cubic B-spline control points.

The model has no SIREN, INR, coordinate MLP, or neural-network parameterization.
The trainable object is a tensor of cubic B-spline control displacements. The
final deformation remains a single cubic B-spline field at control-point
spacing 2. The multistage procedure changes the optimization path; it does not
compose multiple residual deformation fields.

The exact registry candidate is `MR_A_fast_decay`:

```text
method             = direct_cp_8_4_2_be_bspline_analytic_multistage
full control cps   = 8, 4, 2
image downsampling = 4, 2, 1
alpha              = 1000
stage learning rate= 3e-2, 1e-2, 3e-3
early stopping     = adaptive
```

The candidate's registry-level `lr` metadata is `1e-2`, but the explicit
`stage_lr` list above controls the actual optimizer learning rate at each
stage. The stage list therefore takes precedence for this candidate.

## 2. Data and coordinate conventions

### Image layout

The active OASIS loader uses scalar volumes in repository order:

```text
spatial tensor shape = [D, H, W] = [160, 224, 192]
```

The source NIfTI array is converted by the repository's standard
`reorient_lia_to_ras` operation: transpose axes `[0, 2, 1]`, then flip axes 0
and 2, followed by a contiguous copy. The loader verifies the resulting shape
against `(160, 224, 192)`.

Moving and fixed images are jointly min-max normalized. The lower and upper
bounds are computed over both raw images, with a small epsilon in the
denominator, and both normalized arrays are stored as float32.

The default image foreground convention is the raw fixed-image mask:

```text
fixed_image_mask = fixed_raw > 0
```

Segmentations are loaded for Dice and surface metrics. They are not the
default training foreground, sampling domain, or optimization mask.

### Normalized-grid coordinates

The implementation uses PyTorch `grid_sample` coordinates in `[-1, 1]` with
`align_corners=True`. Spatial storage remains Z-Y-X, but the last coordinate
component is ordered X-Y-Z:

```text
grid tensor shape     = [D, H, W, 3]
grid component order  = [x, y, z]
volume storage order  = [z, y, x]
```

The identity grid is made with one `torch.linspace(-1, 1, size)` per spatial
axis. A dense displacement `u[p]` is added to this grid:

```text
phi(p)              = p + u(p)
warped_moving(p)    = moving(phi(p))
```

Thus the optimized field is used as a pull-sampling grid: each output/fixed
location queries the moving image at the displaced normalized coordinate. The
warp is implemented with 3-D `torch.nn.functional.grid_sample` in bilinear
mode for images. Segmentation warps use nearest-neighbor mode during metric
evaluation.

Normalized-grid XYZ displacement is converted to voxel XYZ displacement by

```text
dx_voxel = dx_norm * (W - 1) / 2
dy_voxel = dy_norm * (H - 1) / 2
dz_voxel = dz_norm * (D - 1) / 2
```

For `(D,H,W)=(160,224,192)`, the factors are `(95.5, 111.5, 79.5)` for
`(x,y,z)`.

## 3. Multiresolution design

The fast schedule is:

| Stage | Full cps | Downsample | Domain cps | Loss volume ZYX | Control grid ZYX | Controls | Trainable scalars | LR | Max epochs | Min epochs | Patience |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 4 | 2 | `(40,56,48)` | `(24,32,28)` | 21,504 | 64,512 | `3e-2` | 400 | 40 | 20 |
| 1 | 4 | 2 | 2 | `(80,112,96)` | `(44,60,52)` | 137,280 | 411,840 | `1e-2` | 800 | 80 | 35 |
| 2 | 2 | 1 | 2 | `(160,224,192)` | `(84,116,100)` | 974,400 | 2,923,200 | `3e-3` | 1,300 | 130 | 50 |

The maximum budget is 2,500 optimizer steps per case:

```text
400 + 800 + 1300 = 2500
```

The effective control spacing in a downsampled loss domain is

```text
domain_cps = full_cps / downsample
```

It is 2 at every stage. The image dimensions are computed with ceiling
division, so the downsampled shapes are exactly `(40,56,48)`, `(80,112,96)`,
and `(160,224,192)`.

### Antialiasing and image construction

For downsample factors greater than one, each image is Gaussian blurred before
trilinear downsampling:

```text
sigma        = 0.5 * downsample
kernel radius= 3 * sigma
interpolation= trilinear, align_corners=True
```

The fixed and moving images are prepared independently using the same stage
factor. At factor 1, the original normalized image is used directly, with no
blur.

### Nested control grids

The implementation uses `CP_MARGIN = 4`. For base spacing 8, the control-grid
shape is constructed from the image dimensions and the spacing. Consecutive
stages must satisfy the exact dyadic relation:

```text
fine_cps   = coarse_cps / 2
fine_shape = 2 * (coarse_shape - 4) + 4
```

This produces the control grids in the table. A schedule that violates this
condition is rejected; arbitrary control-grid resizing is not silently used.

## 4. Cubic B-spline interpolation

For scalar coordinate `x`, let `t = |x|`. The implemented cubic basis is
compactly supported on `[-2,2]`:

```text
B(x) = 2/3 + 0.5*t^3 - t^2       for 0 <= t < 1
       (2 - t)^3 / 6             for 1 <= t < 2
       0                         for t >= 2
```

The first derivative is:

```text
B'(x) = (1.5*t - 2) * x          for 0 <= t < 1
        0.5 * (t - 2)^2           for 1 <= t < 2 and x < 0
       -0.5 * (t - 2)^2           for 1 <= t < 2 and x >= 0
        0                         for t >= 2
```

The second derivative is:

```text
B''(x) = 3*t - 2                  for 0 <= t < 1
        -t + 2                    for 1 <= t < 2
         0                        for t >= 2
```

For a control lattice `c[i,j,k]`, the displacement at a spatial location is
the separable tensor-product expansion:

```text
u(z,y,x) = sum_i sum_j sum_k c[i,j,k]
           * B((z-z_i)/cps)
           * B((y-y_j)/cps)
           * B((x-x_k)/cps)
```

Each control point has three displacement components. The external layout is
`[Zc,Yc,Xc,3]`; expansion internally permutes this to
`[1,3,Zc,Yc,Xc]`.

### PyTorch expansion

The one-dimensional kernel is sampled at spacing `1/cps`. Its length is
`4*cps-1`:

| cps | Kernel length | Padding |
|---:|---:|---:|
| 8 | 31 | 15 |
| 4 | 15 | 7 |
| 2 | 7 | 3 |

Expansion applies three grouped `conv_transpose1d` operations, one along each
spatial axis. The transposed convolution uses:

```text
stride  = cps
padding = (kernel_length - 1) / 2
groups  = every independent component/spatial line
```

After the three separable operations, the expanded tensor is cropped to the
requested evaluation shape using slices beginning at `cps`. The result is
returned as a contiguous `[D,H,W,3]` tensor.

This is an exact implementation detail important to acceleration work:
expansion is not a generic interpolation library call, and replacing it with
trilinear interpolation or a differently aligned B-spline routine can change
the represented field and boundary behavior.

## 5. Coarse-to-fine prolongation

Stage 0 starts from an all-zero control tensor. At each later stage, the
previous control tensor is refined by crop-aligned dyadic cubic knot insertion.
For each one-dimensional axis, the inserted coefficients alternate as:

```text
even = 0.5 * (c_i + c_{i+1})
odd  = (c_i + 6*c_{i+1} + c_{i+2}) / 8
```

The operation is applied along z, y, and x. Under the nested-grid relation,
the refined coefficients represent the same continuous B-spline field before
the next optimizer updates them. The implementation can optionally expand
both fields for a preservation diagnostic.

The refinement is not trilinear resizing of the control array. It is exact
dyadic cubic knot insertion for the supported crop alignment.

Each stage creates a new `torch.nn.Parameter` and a new Adam optimizer. Adam
momentum/state is not transferred between stages. The same original moving
image is used at all three stages; only the loss-domain resolution changes.
The method therefore continues optimizing one absolute displacement field and
does not warp the moving image by the previous stage's result.

## 6. Objective and PyTorch training loop

At each stage and epoch, the implementation performs the following sequence:

1. Reshape the flattened `torch.nn.Parameter` into `[Zc,Yc,Xc,3]`.
2. Expand the controls into a dense displacement on the current loss volume.
3. Add the displacement to the stage identity grid.
4. Warp the moving loss volume with 3-D bilinear `grid_sample`.
5. Compute global NCC data loss.
6. Compute analytic cubic B-spline bending energy from derivative expansions.
7. Form the total loss.
8. Run Adam with `zero_grad(set_to_none=True)`, `backward()`, and `step()`.

The exact objective is:

```text
loss = data_loss + alpha * regularizer
alpha = 1000
data_loss = negative global NCC
regularizer = analytic cubic B-spline bending energy
```

### NCC

For the two stage volumes `x1` and `x2`, the implementation computes:

```text
cc  = mean((x1 - mean(x1)) * (x2 - mean(x2)))
std = StableStd(x1) * StableStd(x2)
ncc_loss = -(cc / (std + 1e-10)).mean()
```

The training NCC is global over the full stage image volume. The fixed-image
foreground mask is not passed into this optimizer-loop NCC call. The custom
`StableStd` autograd function uses a detached standard-deviation value while
providing a manually defined gradient.

The DirectCP multistage path uses full-volume tensors and does not use
coordinate minibatching. The active parameter and image tensors are float32
unless an explicitly changed implementation introduces another dtype.

## 7. Analytic B-spline bending energy

The native optimization regularizer is the sampled analytic B-spline bending
energy:

```text
BE(u) = mean over components and stage voxels of
        u_zz^2 + u_yy^2 + u_xx^2
        + 2u_zy^2 + 2u_zx^2 + 2u_yx^2
```

The implementation constructs six derivative fields:

```text
(2,0,0), (0,2,0), (0,0,2),
(1,1,0), (1,0,1), (0,1,1)
```

For derivative order `q` along an axis, the sampled derivative kernel is
scaled by `(1/cps)^q`. Each derivative field is expanded with the same
separable transposed-convolution machinery as the displacement field, with a
derivative kernel substituted on the selected axes.

The optimization call uses full-domain regularization and mean component
reduction. Foreground masks may be used by post-training regularity reports,
but are not passed to this training regularizer.

## 8. Learning-rate and adaptive stopping strategy

The actual decayed stage rates are:

```text
stage 0: 3e-2
stage 1: 1e-2
stage 2: 3e-3
```

No scheduler updates the rate within a stage. The rate changes only when a new
stage and a new Adam optimizer are created.

Adaptive stopping is stage-local. The configuration is:

```text
check interval       = 25 epochs at every stage
minimum epochs       = 40, 80, 130
patience             = 20, 35, 50
relative total limit = 1e-4 at every stage
relative data limit   = 5e-5 at every stage
```

The implementation first ensures that patience is at least the check window:
`patience = max(check_interval, patience)`. At a check point, it compares the
median of the most recent 25 rows against a preceding 25-row median window,
with the configured patience gap between them:

```text
recent_total = median(recent loss window)
previous_total = median(previous loss window)
recent_data = median(recent data_loss window)
previous_data = median(previous data_loss window)

relative_total = (previous_total - recent_total)
                 / max(abs(previous_total), 1e-8)
relative_data = (previous_data - recent_data)
                / max(abs(previous_data), 1e-8)

stop if relative_total < 1e-4 and relative_data < 5e-5
```

The stage must satisfy both the minimum-epoch and available-window
requirements. With a 25-epoch check interval, the first possible practical
checks are at epochs 50, 100, and 150 for stages 0, 1, and 2 respectively,
because checks occur on interval boundaries and the history must contain the
comparison windows.

If the condition is not met, the stage continues to its maximum epoch budget.
The runner records requested epochs, actual epochs, and a structured stop
reason. It does not restore the best historical control tensor after a plateau;
the current control state at the stopping epoch is retained.

## 9. Output and timing semantics

The timing reruns use `output_mode=dvf_only`. The final output is:

```text
file format = NumPy .npy
dtype       = float32
shape       = (160,224,192,3)
units       = normalized-grid XYZ displacement
field       = final cps-2 dense field only
```

Accepted fields must be finite. Intermediate full-resolution fields are not
materialized in the standard timing path unless an explicit diagnostic option
requests them.

The headline timing interval is:

```text
case_load_complete -> dvf_file_ready
```

`case_load_complete` is marked after the fixed/moving images and labels have
been loaded onto the selected device. `dvf_file_ready` is marked after the
final dense displacement has been converted to a CPU float32 NumPy array and
the `.npy` serialization has completed.

Therefore the interval starts after image and label loading is complete and
includes optimization, final dense B-spline expansion, CPU conversion, and
DVF writing. It excludes image loading itself, Dice, Hausdorff/ASD, dense
image similarity, and other metric evaluation, as well as metadata packaging
after the DVF endpoint.

The detailed recorder also exposes:

- device setup time;
- stage setup and loss-volume preparation;
- per-stage optimizer-loop time;
- control-grid reshape;
- B-spline expansion;
- warp/NCC;
- analytic regularizer;
- backward and Adam update;
- final dense expansion;
- CPU DVF preparation and file writing.

Peak GPU memory is reset at the beginning of training and reported as
`torch.cuda.max_memory_allocated()` after the final stage. It is therefore a
training-process allocation maximum, not total device usage by all processes.

For the timing protocol, a cold measurement starts a fresh worker. Warm
measurements reuse the same worker after the cold run, allowing process-level
CUDA context, allocator, kernel, and backend-selection state to persist. A
warm measurement is not a new cold-start measurement.

## 10. CUDA, cuDNN, and reproducibility

The runner receives the requested physical GPU as a string, assigns it to
`CUDA_VISIBLE_DEVICES`, and then selects logical CUDA device 0 inside the
process. Thus a physical GPU selection such as GPU 3 appears internally as
`cuda:0`.

At process setup it sets:

```python
torch.backends.cudnn.benchmark = requested_benchmark
torch.backends.cudnn.deterministic = requested_deterministic
```

The run metadata records both requested flags and effective values observed by
PyTorch, along with PyTorch version, CUDA runtime version, cuDNN version, CUDA
device index, and device name. Torch and NumPy are seeded with the configured
seed, normally seed 1 for the Table-1 protocol.

`deterministic=True` records the cuDNN deterministic setting. It does not by
itself assert that every CUDA operation is globally bitwise deterministic;
non-cuDNN kernels, reductions, library behavior, and execution environment can
still affect reproducibility. The backend metadata and exact Git provenance
must therefore accompany performance comparisons.

## 11. Acceleration-audit targets and invariants

The main cost surfaces exposed by this implementation are:

1. repeated separable B-spline displacement expansion;
2. six analytic derivative-field expansions per optimizer step for bending
   energy;
3. full-volume 3-D `grid_sample` and global NCC;
4. backward through expansion, warp, NCC, and regularizer;
5. stage loss-volume blur/downsampling and control-grid setup;
6. new control tensors and Adam state at stage transitions;
7. final full-resolution dense-field materialization;
8. CUDA synchronization, CPU conversion, and `.npy` writing.

Potential acceleration ideas are hypotheses for audit, not changes made here:

- cache or reuse fixed B-spline kernels;
- reduce temporary allocations in grouped 1-D convolution;
- fuse or otherwise restructure separable expansion;
- combine derivative computations where mathematically equivalent;
- reduce unnecessary synchronization and host-device transfers;
- evaluate `torch.compile` or equivalent graph capture;
- evaluate mixed precision with numerical and stopping-tolerance checks;
- evaluate alternative optimizer implementations only if the optimization path
  is intentionally allowed to change.

Any accelerated implementation should be tested against these invariants:

- identical coordinate ordering and `align_corners=True` convention;
- identical B-spline basis, padding, crop, and knot-insertion alignment;
- equivalent dense fields within a declared numerical tolerance;
- equivalent losses and gradients within a declared tolerance;
- unchanged actual stage stopping behavior unless explicitly intended;
- unchanged final shape, dtype, units, and finite-value checks;
- unchanged alpha, stage rates, stage boundaries, and seed protocol;
- unchanged timing endpoint and exclusion of metric evaluation;
- no unintended change to requested or effective cuDNN/backend settings.

Mixed precision, compilation, kernel fusion, and optimizer changes require
special attention because small numerical changes can alter adaptive stopping
epochs even when the final image appears visually similar.

## 12. Source map

The implementation claims in this document map to these repository sources:

| Topic | Source |
|---|---|
| OASIS loading, normalization, coordinates, warping, B-spline kernels, expansion, derivatives, backend metadata | `scripts/experiments/oasis_alpha_sweep/common.py` |
| CLI flags, multistage control initialization, Adam loop, early stopping, timing, DVF output | `scripts/experiments/oasis_alpha_sweep/run_one.py` |
| Candidate registry and exact `MR_A_fast_decay` values | `scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py` |
| General DirectCP and B-spline implementation summary | `docs/implementation/directcp_bspline_optimization.md` |
| OASIS protocol and method conventions | `docs/03_methods.md` and `docs/standards/oasis_brainmr_protocol.md` |
| Fast-candidate selection and schedule record | `experiments/2026-08-13_oasis_unified_acceleration_sweep/codex_execution_spec.md` |
| GPU-3 timing protocol and validation record | `experiments/2026-09-16_oasis_mrdbscp_cudnn_table1_first10_gpu3/codex_execution_spec.md` |
| Completed three-arm first-10 timing record | `experiments/2026-09-16_oasis_table1_candidates_cudnn_first10_gpu3/technical_log.md` |

## Appendix A. First-10 timing evidence

The following is implementation evidence from the seed-3 shuffled first-10
OASIS diagnostic, not the publication OASIS-100 result. It is included to
provide context for an acceleration audit. Timing is on physical GPU 3 with
the endpoint defined above; Dice is evaluated separately and is not included
in runtime.

Environment recorded by the experiment:

```text
PyTorch = 2.6.0+cu118
CUDA    = 11.8
cuDNN   = 90100
GPU     = NVIDIA RTX 6000 Ada
Git     = 0d16f8f4fdb6fa252b9a858971a8d2d545aed7b2
dirty tree = true
```

| Backend | Cold s | Warm mean +/- SD s | Cold-warm gap s | Actual steps cold / warm mean | Peak GB cold / warm | Dice mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|
| `benchmark=False, deterministic=True` | 23.80 | 9.13 +/- 0.30 | 14.67 | 450 / 437.5 | 1.93 / 1.93 | 0.8061 +/- 0.0417 |
| `benchmark=False, deterministic=False` | 22.48 | 9.25 +/- 0.28 | 13.23 | 450 / 437.5 | 1.93 / 1.93 | 0.8061 +/- 0.0417 |
| `benchmark=True, deterministic=False` | 102.52 | 9.38 +/- 0.85 | 93.14 | 450 / 437.5 | 3.86 / 1.94 | 0.8061 +/- 0.0417 |

These values show that, for this first-10 GPU-3 run, the large paired
benchmark-enabled cold-start overhead was not accompanied by a comparable
warm-time increase. They should not be interpreted as a general hardware
claim or as a replacement for the OASIS-100 publication table. The complete
case-level rows, monitoring logs, retry provenance, and machine-readable
artifacts remain in the external experiment artifact store.

## Appendix B. Validation checklist for an accelerated implementation

Before accepting an acceleration, verify at minimum:

- basis values and derivative values match at support boundaries;
- cps 8, 4, and 2 kernel lengths and padding are unchanged;
- nested control shapes are unchanged;
- zero-control expansion is exactly zero;
- dyadic knot insertion preserves the dense field within tolerance;
- one-step loss and gradient comparisons pass for all three stages;
- fixed-seed stage actual epochs and stop reasons are compared;
- final DVFs are finite float32 `(160,224,192,3)` normalized-grid XYZ arrays;
- backend metadata and Git provenance are recorded;
- timing is still measured from `case_load_complete` through
  `dvf_file_ready`, with evaluation excluded.
