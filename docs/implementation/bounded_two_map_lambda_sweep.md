# Bounded two-map B-spline lambda sweep

P76 preserves the P70 bounded composition numerical core while varying one
explicit diffusion weight across cps8/4/2. Each level has two independent
identity-initialized maps, each with the full stage Adam schedule. Their
latents are normalized-grid XYZ and controls are cap*tanh(latent/cap), with
the validated endpoint-aware0.99 bound. Lambda does not alter the cap.

The chain retains all six elementary displacement interpolants, composed in
pull order Phi_previous o delta_new. Previous parameters are detached but
coordinate gradients through their samplers remain active. This is neither
one prolonged B-spline nor scaling-and-squaring of an SVF.

The data term is raw1-mean(local NCC9 correlation squared). The regularizer
is unmasked forward-valid L2 diffusion of the total composed field, converted
to the current evaluation-grid voxel units. Both maps receive the full lambda;
it is not divided between them. This experiment explicitly overrides the
repository's generic analytic-spline regularization default.

The three levels receive normalized LRs0.03/0.01/0.003 and no LR reductions;
the global map index must not accidentally activate the legacy cps1 schedule
for the fourth map (which is cps4b). Six selected checkpoints are independently
replayed. Cross-runlambda1/P70 differences are reported without demanding
bitwise identical independent optimization.

Timing separates summed optimizer time, complete worker time and independent
audit time. Memory reports allocated and reserved peaks, separately from
live whole-GPU occupancy used for admission. Neither shared GPU timing nor
equal numeric lambdas establish equal compute/constraint strength across
bounded displacement, prolonged displacement, SVF and learned methods.

Source: scripts/experiments/oasis_alpha_sweep/run_oasis_bounded2_842_sweep.py,
audit_oasis_bounded2_842_sweep.py and queue_oasis_bounded2_842_sweep.py.
Protocol/log: experiments/2026-10-15_oasis_bounded2_842_lambda_sweep/.

The accepted reporter independently reads all 70 replayed candidates and
checks complete ten-case groups, all loss histories and saved checkpoints
before writing acceptance. It retains per-label results and paired differences,
including a separate fresh-versus-historical lambda-1 appendix. The figure
adapter replaces only the three-level bounded-two points in C/F; G keeps its
four-level source. Independent figure verification recomputes sample statistics
from exact source rows and checks actual axes and visible uncertainty extents.
Sources: scripts/reports/report_oasis_bounded2_842_sweep.py,
oasis_first10_bounded2_sweep_arc.py and verify_oasis_first10_bounded2_arc.py.
