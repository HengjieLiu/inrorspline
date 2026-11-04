# Matched-prefix cps8421 extension (P80)

P80 isolates the addedcps1 stage by retaining each accepted P79 same-case,
same-family, same-lambda cps842 prefix. The displacement arm refines selected
displacement coefficients; bounded2 retains six maps and optimizes two new
bounded increments; SVF refines selected total velocity, never composes stage
velocities. Historical numerical functions and artifacts are immutable.

New workers use the unchanged loss/Adam schedule with explicit cps1 metadata,
separate prefix provenance and a scoped proof-gated roundoff validator.
Eight-slot durable scheduling separates worker completion from independent
acceptance and defers individual failures. Smokes, replay and memory admission
gate execution. Reporting uses complete100-case groups and matched depth pairs;
newcps1 runtime is distinct from saved prefix timing. Display-only zero-folding
markers never alter numerical data. Full source specification lives under
`experiments/2026-10-16_oasis_bspline8421_matched_prefix/`.

The active entry point is
`scripts/experiments/oasis_alpha_sweep/oasis_bspline8421_diagnostic_recovery.py`
(`--stop-after full`, default `execution_v2`). The first worker/queue sources
remain frozen as the original smoke record. The isolated recovery changes only
the coordinate-autograd context for off-grid diagnostics; the optimization
objects are identical. execution_v2 repeated allnine smokes successfully.

## Foreground-diffusion ARC companion

`scripts/reports/oasis_matched_depth_diffusion_arc.py` adds two read-only
saved-field figures, one per depth. It computes a common final-displacement
metric in native voxel units: squared forward differences averaged over the
three components and over foreground-valid edges on each axis, then equally
averaged over the three axes. Both endpoints must satisfy fixed raw image > 0;
there is no additional crop or lambda multiplier. Full-volume recomputation
is checked against existing canonical diffusion values.

This diagnostic must not be described as the exact optimization loss. Training
diffusion is unmasked; refined SVF additionally penalizes velocity rather than
the final displacement used for all five plotted families. Existing accuracy,
Jacobian metrics and primary plots are unchanged. Each depth uses31 complete
100-case settings, with shared DL rows, common display axes, and marginal
sample-SD ellipses rather than confidence regions.
