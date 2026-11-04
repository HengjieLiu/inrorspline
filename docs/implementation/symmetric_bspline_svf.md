# Symmetric refined B-spline SVF, Phase A

P77 keeps two total cubic B-spline velocity fields, refined independently by
exact knot insertion at cps8/4/2. Intermediate pyramid maps are never composed.
For moving A and fixed B, P1=exp(v1) and P2=exp(v2) sample original stage
images into a common space. Final fixed-to-moving pull is P1 o exp(-v2);
the inverse-direction construction is P2 o exp(-v1). These are retained two-
map chains for off-grid diagnostics and sampled native DVFs for canonical
accuracy/regularity metrics.

Independent parameter tensors represent twice the actual branch velocities,
in normalized XYZ coordinates. This explicit factor one-half preserves the
established 0.03/0.01/0.003 parameter LR convention without inadvertently
doubling the equal/opposite relative-motion step. Adam updates the pair jointly;
the selected checkpoint always contains both branches from the same iteration.

The new NCC9Pair uses historical mean-window kernels, zero statistical padding
and epsilon-clamped variance product, but computes gradients for both images.
The legacy one-input NCC9 caches and detaches the fixed image and must not be
used directly for midpoint optimization. Both images are warped only once
from their original level images; no cached prewarped image enters the loss.

Diffusion remains finite-difference, unmasked, valid-neighbor, component/axis
mean in current loss-grid voxel units. A1 uses 2*(R1+R2); A2 uses R1+R2.
For v1=w/2 and v2=-w/2, A1 equals R(w), whereas A2 equals R(w)/2. This is
an equal-split convention, not universal matching of nonlinear objectives.
The A3 tied midpoint control optimizes one w, using the same regularizer R(w).
At finite integration resolution, composing two seven-step half flows is not
assumed identical to a seven-step full-w integration; sensitivity is reported.

Native-mm magnitude diagnostics use voxel mean of squared vector norm (sum
over XYZ), not control coefficient norms or component means. Common-mode RMS,
mean translations, foreground-union summaries and midpoint out-of-bounds/coverage
are reported beside branch diffusion. Equal RMS need not imply zero common
motion or a geometric midpoint. No magnitude/balance penalty enters Phase A.

Seven-step native integration, border sampling and unclamped coordinates are
unchanged. Independent replay checks each stage's controls, loss history and
both saved full maps. Eight/ten-step re-integrations, 100000 seeded off-grid
cycle/coverage and retained/exported comparisons remain diagnostics. Input
exchange is a separate registration check, not an inverse-consistency claim.

Sources: scripts/experiments/oasis_alpha_sweep/symmetric_bspline_svf.py,
symmetric_svf_diagnostics.py, run_oasis_symmetric_svf.py and
queue_oasis_symmetric_svf.py. Protocol/log:
experiments/2026-10-15_oasis_symmetric_svf_phase_a/.
