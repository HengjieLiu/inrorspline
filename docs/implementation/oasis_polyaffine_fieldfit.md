# OASIS adaptive local-affine field fitting (P81)

P81 fits existing registration endpoints, independently of image similarity
or labels. Its source is `scripts/experiments/oasis_polyaffine_fieldfit/`; the
execution specification and `protocol_v1.3.json` are under
`experiments/2026-11-05_oasis_adaptive_local_affine_fieldfit/`.
This note describes implementation, not observed scientific performance.
Current runtime policy is superseded by the four-GPU scheduling continuation
at the end of this note; scientific implementationv1.3 is unchanged.
Execution completed2026-11-05T16:50:14Z: 1680/1680 jobs and2640 endpoint records.
575/840 main D0 solves remain unresolved; all1680 main SVF final stages reached
their fixed iteration limits. The raw full-box integration-sensitivity failure
and exterior errors remain explicit limitations.

## Coordinates and immutable inputs

`data.py` selects the established seed-3 first10 and exact lambda1 source rows:
historical SITReg, prolonged8→4→2 displacement, two bounded increments per
level (six retained maps), and refined single-SVF with velocity diffusion.
It records source/report hashes and exports independent canonical copies.

Canonical fields have array shape `[D,H,W,3]` and **world-RAS millimetre vector
components**. Scalar reorientation uses the legacy transpose/flips with an
explicitly updated affine. SITReg's saved channel-first zyx voxel field and
the splines' channel-last normalized-grid xyz fields use different adapters.
PyTorch grid xyz denotes W/H/D, not world R/A/S. The physical pullback map is
`fixed_world_point + displacement_mm`; fixed and moving image geometry are
validated before conversion. Foreground is the fixed raw image-positive mask.

## Shared dictionaries and native rendering

`basis.py` exposes scalar features `[N,M]` and coefficients `[M,3]`, so actual
motion parameters are `3*M`. Seven families cover clamped cubic B-splines,
normalized Gaussian translation/affine, and uniform/adaptive translation or
affine trees. Gaussian translation and translation trees contain a shared
affine root; one Gaussian weight is removed to eliminate constant redundancy.
Cubic affine reproduction uses Greville control values and includes boundaries.

Trees use logistic partition-of-unity gates represented by hierarchical
contrasts. A new zero contrast preserves the field and all existing feature
columns. Uniform refinement selects large boxes; adaptive refinement scores
fixed-sample endpoint residuals. Both split the longest physical axis and use
deterministic node-ID ties. Counts include all motion coefficients and record
tree/lattice metadata separately. Geometry is fixed during each coefficient
solve; split selection is an additional optimization operation.
Each production split checks up to512 analytic fit-sample values, supported
by zero-contrast inheritance and generic/separable parity unit tests. It does
not compute a full native dense/renderer difference at every transition.

For diagonal physical grid affines, every basis feature separates into three
one-dimensional factors. Repeated tree gate paths share factor columns and
map into a compact coefficient tensor. `dense_forward`/`dense_adjoint` evaluate
that tensor through contractions, retaining full logistic/Gaussian tails.
`numerics.py` dispatches to this path through a custom coefficient autograd
function. Unsupported geometry or an excessively large factor core falls back
to chunked features and an adjoint that recomputes features. Neither path
retains a native-volume-by-parameter design matrix. Float64 forward/adjoint
and float32 generic/separable parity are explicitly tested; summation order
can produce small floating-point differences.

## Solvers and transformation renderers

D0 uses `Id+u`. `solve_d0` runs float64, column-scaled SciPy LSQR independently
for the three world-vector components, with a matrix-free operator and a
bounded optional cache of sampled features. It records residuals, normal
residuals, condition estimates, termination and actual iterations. An
unresolved solve remains labelled unresolved rather than an attained capacity
optimum. The physical endpoint loss is mean squared vector error in mm²;
there is no deformation regularizer in this fitting objective.

Version1.3 uses shared `conditioning_scales` for LSQR, nonlinear Adam and
sampled Adam/L-BFGS diagnostics: feature RMS on a deterministic endpoint-
including32³ lattice in the declared physical box, floored at0.01. It records
domain, sample count, RMS/scale ranges, floored-column count and scale hash.
The box lattice only conditions coefficients; the foreground fit points,
weights and endpoint objective are unchanged. No background loss, ridge,
coefficient penalty or clipping is added. For an ill-conditioned or
underdetermined system, scaling and finite stopping tolerances influence the
implicit scaled-norm solution selection and extrapolation; warm starts can
retain nullspace components. This is not a unique-solution or stability proof.

The superseded v2 foreground-only scales produced approximately1e11
coefficients, large out-of-ROI fields and divergent SVF initialization during
actual optimization despite passing short resource smokes. Its queue was
stopped and artifacts retained. v3's production-limit LS and actual multistep
zero/global_affine_ls SVF checks were reviewed under the recorded foreground
criterion before launch; this does not establish full-box integration accuracy.
With corrected scaling, a cap768 D0 pilot reached its LS stopping criterion
yet produced approximately14,322mm outside the ROI; cap1536/3072 remained
unresolved. Full-box scaling does not impose a full-box field bound, and
these pilot outputs are not evidence of attained high-budget capacity.

SVF renders the same dictionary as velocity using native-grid
scaling-and-squaring, trilinear border sampling and seven primary squarings.
Coarser optimization grids preserve endpoint-aligned physical geometry;
final refinement and export use the native grid. Adam differentiates through
integration with coefficient-column scaling. Both `zero` and
`global_affine_ls` initialization trials are retained for every family and
frozen replay. `initialization.py` fits a12-parameter affine displacement on
the same ROI samples and embeds it exactly with `coefficients_for_affine`.
It interprets that displacement as initial velocity, without a matrix
logarithm; its integrated endpoint need not equal the D0 affine map. Subsequent
optimization uses the full selected basis. This avoids using an unconstrained
local-LS extrapolation as the SVF start and changes neither D0 nor D1 local LS.
Eight-step reintegration is a diagnostic and does not replace the target or
primary seven-step export.
The existing sensitivity threshold uses the full raw-positive foreground
vector RMSE, `max(0.001,0.05*sqrt(native_fit_MSE_mm2))`. The multistep raw
pilot used an additional exploratory0.05mm full-box threshold and failed it,
despite improving fit and low foreground differences. Its raw failure and
large exterior differences remain preserved. A separate frozen-coefficient
review passed the existing foreground criterion without changing fitting
or the renderer; this cannot establish full-box integration accuracy.
At checkpoints, training/validation loss above its respective
`min(1e6,max(100,100*initial_MSE))` mm² limit marks the attempt failed. This
diagnostic abort preserves evidence and does not change or clip the objective.

D1 fits the rendered bounded field. Its radial rescaling preserves constant
translation and uses a full-cell bound on the **displacement Jacobian Du**,
not `Dphi=I+Du`. Every cell-sided corner and induced infinity-norm row sum is
checked in physical coordinates. The selected bound witness supplies the
piecewise subgradient; the final float32 field is checked again in float64.
This sufficient condition is for a trilinear map on the declared box and is
not a claim of a global smooth inverse outside that domain.

## Matrix, queue and failure semantics

The full main matrix has840 D0 configurations and840 SVF configurations over
40 targets, seven families and three parameter caps. Two SVF initializations
produce1680 main SVF trials. Additional work contains40 global-affine anchors,
24 D1 pilot configurations,24 new topology-replay configurations/32 trials,
and24 frozen-geometry solver diagnostics. Tree paths share budget checkpoints;
the resulting queue has1680 worker jobs, distinct from endpoint/trial counts.
Translation same-K SVF supplements are native renderings of coarse growth
states, without the main budget's half/native refinement. They are stage-tagged
trajectory diagnostics, not convergence-matched capacity endpoints.

`launch.py` binds the input audit, source metric replay, native resource
profiles, serialized worker smoke, exact source hashes and frozen configuration.
The v3 launch contract additionally requires a hash-bound actual multistep
stability receipt; a one-update memory probe is insufficient.
`controller.py` admits jobs only to physical GPU0, with at most eight active
jobs and measured memory reservations/headroom. It also checks host RAM and
disk space. Workers set cuDNN benchmark false/deterministic true and TF32 off;
this does not imply bitwise-deterministic sampling backward.

Each attempt has durable launch intent, PID/start-time/UID identity, progress,
completion or failure records and immutable outputs. Individual exceptions,
timeouts or OOMs are recorded and independent work continues. Missing failed
dependencies become explicit dependency failures. Watchdogs only signal the
identity-verified worker process group; no automatic retry loop is used.
Infrastructure admission has a deadline and unresolved work is marked deferred.
Queue completion and scientific acceptance remain separate concepts.

## Evaluation and limitations

`metrics.py` reloads exported fields and measures endpoint error on foreground,
full FOV and a10mm ring. Physical central derivatives use a common one-voxel
interior and eroded ROI, with slab-sized Jacobian arrays. This stencil/trim is
documented separately from the legacy two-voxel trim. Determinant distributions,
new/removed folding and positive-only log-Jacobian errors retain coverage.
Sampled zero folding is not a continuous invertibility proof.

Image/label warps explicitly use zero padding and `align_corners=True`.
Image fidelity uses the reference in-bounds domain and retains prediction
out-of-bounds values; coverage is reported. Labels are evaluation-only.
Dice, HD95 and pooled-surface-area symmetric ASD use explicit absent-label
records and common-label paired retention. Rings/surface distances require
orthogonal image geometry; unsupported shear fails instead of approximating
physical distances silently.

`diagnostics.py` supplies full-foreground best-global-affine residual
normalization, analytic local-slope/blending derivative decomposition,
bidirectional exported inverse residuals with intermediate-domain coverage,
and sampled Adam/L-BFGS versus LS diagnostics. The latter reports Adam's
training loss and the final post-L-BFGS coefficients; it does not claim an
evaluated native-grid Adam gap. L-BFGS has a hard closure-evaluation cap.

The [analysis completion ledger](../../experiments/2026-11-05_oasis_adaptive_local_affine_fieldfit/analysis_completion_ledger.md)
tracks outstanding work separately from fitting coverage. Local Taylor
residuals at8/16/32mm, derivative variation, neighboring-affine disagreement
and transition residual/fold maps are not yet implemented. The current
reporter produces coverage and selected mean/SD summaries; final paired
differences, median/IQR, win/tie/loss, scientific figures and an executed
review notebook remain post-fit work. Conditional native source-velocity
replay is unresolved, and other conditional branches are not baseline jobs.

The first10 were repeatedly explored, and source fields are algorithm outputs,
not ground-truth deformations. Fixed source lambda1 does not imply equal
regularity across representations. Native velocity checkpoint replay and
stronger optional diagnostics remain separate from endpoint fitting. All
generated data, reports, fields and executed notebooks belong under
`${ARTIFACT_ROOT}`, with source/configuration snapshots bound to each run.

## Scheduling continuation

The [multi-GPU runtime addendum](../../experiments/2026-11-05_oasis_adaptive_local_affine_fieldfit/scheduler_v1.md)
supersedes this note's original GPU0/max8/no-retry scheduling description.
Separate scheduler sources preserve the frozen fitting implementation and adopt
live workers. GPU0/1 allow16 workers each; GPU2/3 allow4 each, subject to
independent memory tiers and a shared queue. CUDA OOM retries preserve attempts
and published checkpoints, increase reservations and return to any eligible GPU.
Busy devices are polled every60 seconds without blocking other devices.
