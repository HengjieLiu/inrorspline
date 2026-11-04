# Symmetric SVF amplitude/balance ablation

Phase B is isolated in symmetric_bspline_svf_phase_b.py, its worker/controller
and independent tests. Phase-A sources are imported read-only and source-hashed.
The objective retains two-gradient midpoint NCC9 and coefficient2 summed velocity
diffusion. It adds alpha*(E1+E2) and beta*(sqrt(E1+1e-8)-sqrt(E2+1e-8))^2.
E is full native voxel mean of squared vector norm of each actual velocity in
mm; no mask or coefficient-space shortcut. Native 1mm OASIS geometry is inherited
from audited loaders. Neither new term uses the downsampled loss-grid voxel scale.

The optimizer reports unweighted/weighted diffusion, amplitude and balance for
every joint state. Best selection uses their full sum, including initialization.
Independent replay verifies selected terms, all history arithmetic, fields and
canonical metrics. Alpha=beta=0 preserves the historical computation path.

The approved selection graph is 15 initial fits ->9 combined neighbors ->4
diagnostic registrations. Every fit starts from identity; only final metrics
select coefficient neighbors, not optimization epochs. Diagnostic swaps/repeat
are pinned to their source worker's physical GPU. Immutable plan files record
source summary hashes. Queue preserves individual failures, defers one OOM retry,
and requires complete predecessor groups before choosing later settings.

Shared and opposing velocity RMS are computed directly from (v1+v2)/2 and
(v1-v2)/2, including full-volume and original raw-foreground-union summaries.
Input exchange and same-order repeat discrepancies are reported separately from
within-fit inverse-cycle errors. Equal RMS does not prove a centered midpoint.
Amplitude can suppress useful relative motion; it is not a common-motion-only
penalty. No NCC support correction is included in this ablation.
