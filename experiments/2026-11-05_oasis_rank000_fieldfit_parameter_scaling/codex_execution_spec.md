# P82: rank0 full-box field representation scaling

User-approved 2026-11-05. Two independent execution arms, native and joint
multiresolution, fit the seven P81 direct-DVF families to rank0 original SITReg
and unbounded842 lambda1. Pair: fixed0333, moving0115, seed3 shuffled first10.
No SVF, image registration, foreground training mask, or regularizer.

## Matrix and immutable inputs

Six scalar caps:3072,24576,196608,1572864,12582912,20643840. Two targets x seven
families x six caps x two arms =168 endpoints. Actual counts, not cap labels,
are the plot abscissa. P81 canonical field and original source hashes bind inputs.
Final shape160x224x192, world RAS millimeters, fixed-to-moving pullback.

Source lives in `scripts/experiments/oasis_fieldfit_scaling/`; configuration is
`config.json`. Run outputs go to `${ARTIFACT_ROOT}/runs/oasis_rank000_fieldfit_parameter_scaling_v1/<execution-id>/`.
Review outputs and executed notebook go beneath that run's external `reports/`.
Previous P81 implementation and artifacts are not modified.

## Representation and exact evaluation

Preserve P81 global affine roots, Gaussian sigma/h=.85, sigmoid gate width=.12,
parameter counts and partition rules. Trees permit min child .25mm/depth32.
Full tensor grid contractions and shared-axis tree contractions have exact
adjoints and full-domain column L2 preconditioning. Bounded dense tree cores
fall back to blocked separable rank contractions. No tail pruning or low-rank
approximation. Compact structured arrays replace Python node dictionaries.
Final fixed lattices use the P81 maximum-budget/anisotropy<=1.6 selection rule;
the two arms share identical final dictionaries.

## Optimization and time accounting

Each endpoint starts independently. Native grids start at zero; native trees
grow from a global-affine root. No borrowed lower-budget fit work. Full-box
vector MSE uses all native centers; evaluation additionally reports image>0
foreground and its complement. Float64 LSQR has atol=btol=1e-6,conlim=1e12,
max20000/component, with true residual/stationarity checks. Intermediate tree
growth solves use at most8 iterations; final geometry receives remaining time.
Best finite objective checkpoints are retained; no capacity-optimum claim for
timeouts or condition/iteration stops.

Active fitting, construction, conditioning, growth, checkpoint writes and
transfers share1200seconds. Queue waiting, input loading and final evaluation
are recorded separately. Deadlines are checked at operator/block boundaries;
in-flight kernels cannot be preempted and any measured overshoot is reported.
OOM retries preserve accumulated time and recover saved coefficients/topology.

## Joint multiresolution

Quarter/half/native target shapes40x56x48,80x112x96,160x224x192 have cumulative
deadlines120/360/1200seconds; unused time carries forward. Build coarse targets
directly from native: Gaussian sigma=.5factor,radius=ceil(3sigma),replicate
padding, then trilinear align_corners interpolation. Preserve physical endpoints
and vector millimeter units; native target remains unsmoothed.

Spline intermediate knots retain every fourth/second final boundary plus both
endpoints, with clamped degree3 boundaries. Exact Boehm knot insertion preserves
the represented field. Gaussian per-axis counts are max(2,ceil((n-1)/f)+1),
uniform across the box; transfer fits the previous continuous field on the next
grid with a maximum60-second LS projection charged to the receiving stage.
Projection error/stopping reason are recorded. Final Gaussian bases stay exact.
Tree leaf goals ceil(K/64),ceil(K/8),K; zero-detail splits preserve the field.
Adaptive scoring sums current-grid weighted squared residual over the full box.
Stage timeouts advance with the available model; unachieved final counts are
budget_not_reached and are not plotted as successful requested-cap endpoints.

## Execution and reporting

Configured Docker, current artifact-writing UID/GID, benchmarkFalse,
deterministicTrue,TF32False. Freeze source/configuration hashes. GPU0/1 maxima16,
GPU2/3 maxima4; per-admission memory checks, tiers16/8/4/2/1, host reservations,
cross-GPU dispatch and60-second unavailable-GPU polling. Record OOM attempts,
halve rank tiles, increase reservations, requeue without resetting the timer.
Other failures remain terminal while independent jobs continue.

Preflight requires coordinate/hash checks, original cps2 checkpoint replay
RMSE<1e-4mm, float64 operator/adjoint parity1e-8, exact-transfer tests, Gaussian
projection and rank-deficient LS diagnostics, pyramid unit checks, timeout/
serialization/requeue tests, native memory profiles and worker smokes.

Final deliverables:2x2 seven-family curves (target rows, execution-arm columns),
paired arm comparisons, foreground curves, convergence/time/transfer tables,
matched central error slices, CSV/JSON,PNG/PDF and executed notebook. Log axes,
filled converged/hollow unresolved markers, explicit missing points and gaps.
One image pair, no population inference. Every168 identity receives a disposition.
