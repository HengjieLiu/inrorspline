---
doc_type: implementation_explanation
status: active
---

# Jacobian metrics: central differences, analytic splines, folding, and NDV

This note explains the five regularity columns used in the OASIS FD/analytic
regularizer study. It records the mathematical meaning, the actual repository
conventions, and how to interpret apparently conflicting measurements.

The key distinction is that **central differences and analytic derivatives are
two ways of differentiating a deformation**, whereas **folding percentage and
NDV summarize different aspects of invalid geometry**. Training and evaluation
choices must also be kept separate.

## 1. What "canonical" means in this repository

"Canonical" denotes the repository's established evaluation convention. For
these Jacobian columns, that convention uses central finite differences of the
sampled dense displacement. It does not mean an exact derivative, ground truth,
or a universally accepted standard.

For explanatory tables, prefer these explicit labels:

| Earlier label | More explicit label |
|---|---|
| Canonical SD(logJ) | Central-difference positive-J SD(logJ) |
| Analytic SD(logJ) | Spline-analytic positive-J SD(logJ) |
| Central-difference folds % | Central-difference J <= 0 (%) |
| Analytic folds % | Spline-analytic J <= 0 at sampled locations (%) |
| NDV index | Normalized discrete non-diffeomorphic-volume index |

These naming suggestions do not rename existing output keys or alter any
historical metric definition.

There are two independent choices:

- **Training backend:** which derivative definition the active regularizer uses.
- **Evaluation convention:** which derivative definition measures the saved field.

Every arm in the study receives both Jacobian evaluations. A field trained with
FD-KL can therefore have an analytic folding measurement. "Analytic folds" is
not a column reserved for analytically trained methods.

## 2. The Jacobian describes local volume and orientation

For displacement u and transform phi,

\[
\phi(x)=x+u(x),\qquad
J(x)=\det D\phi(x)=\det\bigl(I+Du(x)\bigr).
\]

Use det(I + Du), not det(Du). Identity has u=0 and J=1.

| J | Interpretation under the measured map |
|---:|---|
| 1 | Local volume preserved |
| 2 | Local volume multiplied by two |
| 0.5 | Local volume halved |
| 0 | Singular first-order map / local volume collapse |
| Negative | Local orientation reversal |

This study intends orientation-preserving registration, so it flags J <= 0 as
folding or collapse. Negative J means orientation reversal; this local sign
test alone is not a complete characterization of global one-to-one behavior.

Map direction matters. The image is warped from moving to fixed, but the pull
sampling map phi sends a fixed-grid location to a moving-image query location:
warped_moving(x) = moving(phi(x)). Interpret volume ratios in that map's
direction; do not identify expansion/compression with a biological tissue change
without stating the direction.

## 3. Central-difference versus analytic Jacobians

### Central differences: differentiate neighboring dense samples

First sample u on the voxel grid. Approximate an x derivative by

\[
\frac{\partial u}{\partial x}(x)
\approx \frac{u(x+h)-u(x-h)}{2h}.
\]

The native-grid calculation uses voxel-coordinate displacement and h=1 voxel.
The derivative averages change across a two-voxel interval. Assemble Du from
the three axis derivatives, add I, and take its determinant to obtain J_FD.

This convention can be applied consistently to exported dense fields from
different methods. Its value depends on sampling resolution, stencil and units;
it need not equal the local derivative of the underlying continuous model.

### Analytic derivatives: differentiate the spline formula

For a continuous cubic B-spline field,

\[
u(x)=\sum_i c_i B_i(x)
\quad\Longrightarrow\quad
\frac{\partial u}{\partial x}(x)
=\sum_i c_i\frac{\partial B_i(x)}{\partial x}.
\]

The implementation uses the tensor-product cubic basis and its exact first
derivatives, with coordinate/component scaling. This gives J_analytic for the
represented spline, up to floating-point error, without finite-differencing the
exported dense samples. Pyramid derivatives additionally require the chain
factors for endpoint-aligned native-to-stage coordinates.

The derivative formula is analytic, but the reported statistic is still sampled:
the current evaluator checks voxel centers. It does not search continuously
between those centers or certify all points in a spline cell.

| Property | Central difference | Spline analytic |
|---|---|---|
| Input | Dense displacement samples | Control coefficients and spline basis |
| Derivative | Numerical approximation | Direct derivative of represented spline |
| Describes | Behavior inferred from grid samples | Local behavior of continuous spline |
| Available for arbitrary dense baselines | Yes, under a fixed convention | Only if a continuous representation is available or explicitly defined |
| Sensitive to evaluation locations | Yes | Yes: analytic differentiation does not remove sampling limits |

For this study, both determinant arrays use the same foreground and crop. Their
differences are not caused by selecting different image pairs or training two
different fields for the two columns. The same saved deformation is measured
twice.

### A central-difference blind spot

Consider a one-dimensional example:

| Original coordinate | -1 | 0 | 1 |
|---|---:|---:|---:|
| Mapped coordinate phi | -1 | 2 | 1 |

The central derivative at coordinate 0 is

\[
\frac{\phi(1)-\phi(-1)}{2}=\frac{1-(-1)}{2}=1.
\]

It appears valid, but the interval from 0 to 1 maps backward from 2 to 1.
The central derivative at the middle point does not use that point's mapped
value. Any differentiable interpolant through these values must have a negative
derivative somewhere in that interval. An analytic check can reveal it if the
evaluation locations include the problematic region; checking only voxel
centers can still miss a between-center reversal.

Related failures motivate multi-configuration digital-deformation tests in
[Liu et al., On Finite Difference Jacobian Computation in Deformable Image Registration](https://doi.org/10.1007/s11263-024-02047-1).

## 4. SD(logJ): variation among positive determinants

Let Omega_2 be the evaluated foreground after the two-voxel border crop. For
either determinant definition J, the evaluator computes

\[
S_J=\{x\in\Omega_2:J(x)>0\},\qquad
\mathrm{SDlogJ}=\operatorname{SD}_{x\in S_J}\bigl(\log J(x)\bigr).
\]

- **Central-difference SD(logJ)** inserts J_FD into this statistic.
- **Analytic SD(logJ)** inserts J_analytic into the same statistic.
- Log is the natural logarithm. Within each volume, the population standard
  deviation is used: NumPy ddof=0 / Torch unbiased=False.
- J <= 0 is excluded, not clamped, replaced by its absolute value, or passed
  through the KL-C1 continuation. Report positive-J coverage and folding beside it.
- No positive determinants gives an undefined SD, not zero. Very small positive
  support makes the statistic unrepresentative even if its value is small.
- Different determinant definitions can select different positive subsets in
  the same foreground, contributing to differences between the two SD values.

SD(logJ) measures **variation of local volume scale**, not closeness to J=1.
Uniform J=2 has SD(logJ)=0 despite a twofold volume scale everywhere. Identity
and volume-preserving shear can also have zero SD(logJ). A low value does not
establish correct correspondence, small displacement, low shear, or invertibility.

This also separates the training loss from the metric. Raw Leow density
(J-1)log(J) penalizes uniform J=2, whereas SD(logJ) does not. KL-C1 extends that
training density below epsilon; the evaluation SD never uses that surrogate.

In cohort tables, compute each case's metric first, then summarize the ten case
values. A reported across-case sample SD uses ddof=1. Do not confuse it with the
within-volume population SD that defines SD(logJ), or pool all cases' voxels.

## 5. Folding percentage: frequency of failed sampled Jacobians

For the chosen determinant convention,

\[
F_J=100\,\frac{1}{|\Omega_2|}
\sum_{x\in\Omega_2}\mathbf{1}\{J(x)\le0\}.
\]

Both negative determinants and exact zeros are counted. Each failed location
has equal weight: J=-0.001 and J=-10 each contribute one to the count.

The central-difference and analytic folding columns differ only in the J used
for this calculation. They are not different physical units or different
thresholds. Their values describe sampled locations, not an exact continuous
percentage of the whole domain that is folded.

## 6. NDV: a discrete negative-volume quantity

NDV denotes non-diffeomorphic volume. The repository computes it from the dense
sampled transform using ten signed determinant configurations: eight combinations
of forward/backward axis differences and two additional diagonal configurations.
They describe local tetrahedral geometry. Negative oriented-volume contributions
are accumulated, rather than counting only their signs. See the
[primary NDV paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11870676/).

### Exact normalization used by this repository

Let D_k(p) be one of the ten determinants returned by
calculate_jacobian_determinants, and let m(p) be the cropped foreground mask.
For the default threshold=0, the code implements

\[
V^-_{\mathrm{discrete}}
=\frac{1}{12}\sum_{p\in\Omega_1}m(p)
  \sum_{k=1}^{10}\max\{-D_k(p),0\},
\]

\[
\mathrm{NDV\ index}
=\frac{V^-_{\mathrm{discrete}}}{N_1},\qquad
N_1=\sum_{p\in\Omega_1}m(p).
\]

Here Omega_1 is the one-voxel-cropped domain. The factor 1/12 is the code's
0.5 weighting times the tetrahedral factor 1/6. Use the returned D_k values
with this weighting; they are not ten interchangeable copies of the central
Jacobian. In unit voxel coordinates, N_1 is the reference ROI volume in voxel
units. The foreground mask weights stencil centers; tetrahedra are not clipped
exactly against the foreground boundary.

The normalized index is stored as ndv_frac_fg, but **it is not the fraction of
foreground voxels that fold**. Its magnitude depends on inverted volumes and
can exceed one. A value0.02 does not mean that2% of foreground voxels are folded.
It is also not an exact unique union volume of all physically overlapping regions.

At exactly D_k=0, the negative-volume contribution is zero. Thus NDV=0 does not
exclude degenerate zero-volume configurations. Together with sampling and
representation assumptions, this prevents using zero NDV as a general validity
certificate.

### Comparison with analytic folding

| Property | Analytic folding percentage | NDV index |
|---|---|---|
| Quantity summarized | Frequency of J_analytic <= 0 at sampled points | Weighted negative volumes of discrete tetrahedral configurations |
| Representation | Continuous B-spline, evaluated at selected locations | Dense sampled transform and its discrete neighboring geometry |
| Role of magnitude | No extra weight for a more negative J | Larger negative-volume contributions increase the index |
| Exact zero | Counted as a failed Jacobian | Adds no negative volume |
| Numerical range | 0-100% | Nonnegative; not necessarily bounded by1 |
| Current border crop | Two voxels | One voxel |

Many small local reversals and a few severe inversions need not have the same
ranking under a count and a volume measure. Differences also reflect distinct
representations, sampled locations, stencils and boundary supports. There is no
fixed conversion between analytic folding percentage and NDV.

A continuous negative-volume analogue would be

\[
V^-_{\mathrm{analytic}}
=\frac{1}{|\Omega|}\int_\Omega
 \max\{-J_{\mathrm{analytic}}(x),0\}\,dx.
\]

This differs from integrating the indicator of J_analytic <= 0, which measures
the extent of failed source locations. **The analytic negative-volume integral
above is not the NDV currently reported.** The current NDV uses the discrete
construction and normalization described earlier, not a numerical integral of
the continuous B-spline determinant.

## 7. Interpretation of the KL-C1 example

The accepted first10 study provides this illustrative comparison at weight0.03.
Values are rounded means of per-case foreground metrics, not pooled voxel counts.

| Training arm | Central-difference folds % | Analytic folds % | NDV index |
|---|---:|---:|---:|
| KL-C1 FD | 0.430 | 4.021 | 0.03918 |
| KL-C1 analytic | 0.507 | 0.553 | 0.02028 |

Central folding alone would slightly favor FD-KL. The analytic derivative finds
substantially more failed sampled locations in that same field, and NDV finds
more inverted discrete volume. Analytic KL makes the two folding estimates much
more consistent and lowers NDV by approximately48% relative to FD-KL.

For FD-KL, canonical SD(logJ)=0.7162 also looks smaller than diffusion FD's0.7628,
but analytic SD(logJ) reverses that ordering:0.8735 versus0.8078. NDV is likewise
higher for FD-KL:0.03918 versus0.01590. The apparent gain therefore depends on
the measurement convention; it is not a uniform geometric improvement.

These observations support checking stencil sensitivity. They do not by
themselves prove a particular spatial failure mechanism or establish validity
between analytic evaluation points. Neither zero sampled folding nor zero NDV
alone proves global orientation-preserving diffeomorphism.

The full generated comparisons remain external under
`${ARTIFACT_ROOT}/repo_experiments/oasis_first10_regularizer_fd_analytic_v1/oasis_reg6_20261110T0917Z_v2/paired_review_v3/`.
This implementation note retains an explanatory example, not a replacement for
the external result tables or their provenance.

## 8. Reporting practice and code map

Keep the established central-difference metrics for historical comparability.
For B-spline models, add analytic Jacobian metrics to assess the represented
continuous field. Use NDV as a complementary discrete-volume check. Report
positive-J coverage, accuracy and label eligibility alongside regularity.

When comparing results, keep map direction, physical/voxel coordinate conversion,
evaluation resolution, foreground rule, border crop and reduction convention
explicit. In OASIS, foreground is fixed raw image>0, not segmentation>0.

| Report column | Output key / convention |
|---|---|
| Canonical foreground SD(logJ) | log_jdet_std_fg; population SD over J_FD>0 |
| Analytic foreground SD(logJ) | analytic_jacobian_fg_std_logj; population SD over J_analytic>0 |
| Central foreground folding % | jdet_le0_pct_fg; already stored as a percentage |
| Analytic foreground folding % | 100 * analytic_jacobian_fg_fraction_le_zero; stored value is a fraction |
| Foreground NDV index | ndv_frac_fg; dimensionless normalized negative-volume index |

Relevant source locations:

- [common.py](../../scripts/experiments/oasis_alpha_sweep/common.py):
  jacobian_determinant_zyx, log_std_positive and jacobian_metrics_voxel define
  the established central/crop2 metrics. calculate_jacobian_determinants,
  calculate_non_diffeomorphic_volume_map and ndv_metrics_voxel define NDV.
- [losses.py](../../scripts/experiments/oasis_regularizer_sweep/losses.py):
  fd_jacobian is the differentiable central counterpart; Derivatives.jacobian
  differentiates the cubic basis with the required coordinate scaling.
- [worker.py](../../scripts/experiments/oasis_regularizer_sweep/worker.py):
  region_stats and evaluate_one apply both determinant conventions, foreground,
  positive-J reduction and canonical parity checks to each saved candidate.
- [Experiment implementation note](oasis_regularizer_fd_analytic_sweep.md) and
  [experiment specification](../../experiments/2026-11-10_oasis_regularizer_fd_analytic_sweep/codex_execution_spec.md)
  define the training arms and their scope.

The source metric definitions remain unchanged by this document.
