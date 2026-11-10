# OASIS rank0 full-box parameter scaling (P82)

P82 is an isolated direct-DVF reconstruction experiment, not image registration.
It compares native fitting and joint target/model refinement on the same final
fixed dictionaries. The seven families inherit P81's mathematical definitions,
but the objective now covers the entire native image box. Old foreground-only
errors are not comparable points on the new curves.

The implementation in `scripts/experiments/oasis_fieldfit_scaling/` uses exact
tensor contractions for B-spline/Gaussian grids. Tree gates are represented as
products of one-dimensional factors; equal axis histories share factors. A
bounded coefficient tensor core uses exact scatter-add/gather; larger cores
use rank-block contractions. This changes evaluation order, not the basis, and
does not prune Gaussian or sigmoid tails. Structured node arrays store tree
geometry. Scalar parameter accounting excludes discrete topology storage.

The float64 LSQR operator uses exact column L2 norms over its current full grid.
Forward/adjoint and dense least-squares contracts check the implementation.
Separate vector-component stopping criteria are evaluated against explicit
residuals and scaled normal residuals. Atol/btol1e-6 are numerical stopping
tolerances, not a guarantee of scientific representation optimality.

Joint refinement builds quarter/half targets directly from native vectors with
antialiasing and physical-box-preserving interpolation; vector values remain
millimeters. B-spline knot subsets are nested, with exact Boehm insertion.
Gaussian transfer is a <=60-second LS projection and is explicitly approximate
initialization. Trees inherit through zero detail coefficients, with adaptive
scores based on current-resolution full-box residuals. No coefficient fitting
work is borrowed from another requested endpoint.

Each endpoint has1200 active seconds, including all intermediate work. Runtime
checks occur at operator boundaries, with in-flight-kernel overshoot reported.
Timeouts, missing requested geometry, numerical failure and resource limits are
separate outcomes. Completed worker execution does not imply a converged solve.

Specification, configuration and output-free notebook:
`experiments/2026-11-05_oasis_rank000_fieldfit_parameter_scaling/`.
Generated reports/notebooks are external under the run's `reports/` directory.
