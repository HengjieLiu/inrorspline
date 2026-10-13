# Full-volume SIREN NCC9/diffusion variant

Status: **cancelled, inactive implementation**. The user requires the historical
10,000 sampled-coordinate IDIR update, not this full-volume training domain.
All matching workers were stopped and active generated artifacts cleared into
recoverable trash. Do not resume this runner or reuse its results. The current
sampled replan is at the top of the experiment execution specification and
awaits an explicit loss-definition choice and approval.

`idir_ncc_diffusion.py` implements an isolated method identity
`idir_siren_fullres_ncc9_voxel_diffusion`, not a modification of the original
sampled-global-NCC/analytic-BE IDIR method. The archived full-volume attempt is
specified in `experiments/2026-10-10_oasis_idir_omega_ncc9_diffusion/`.

Native SIREN initialization is preserved: first-layer weights uniform±1/fan-in;
all subsequent layers including the final linear layer use uniform bounds
sqrt(6/fan-in)/omega; biases retain nn.Linear initialization. Omega multiplies
every hidden sine input. Inputs/outputs are normalized XYZ; no Fourier features.

The full-resolution raw-image objective reuses P43's exact NCC_vfa9 arithmetic
and forward-valid voxel-field diffusion with no masks, BE, sampling-domain
shortcut or lambda normalization. SIREN weights alone receive Adam updates.

Memory-bounded two-pass backprop first assembles the entire field, obtains its
full-image loss gradient, then accumulates network vector-Jacobian products in
coordinate chunks. Cross-chunk image neighborhoods and diffusion edges remain
in the single full-field objective. Compare against a monolithic small-grid
graph for values, parameter gradients and one Adam step before full execution.

Each fit uses fresh seed1, Adam1e-4 and2500updates with no schedule. Primary
metrics use terminal2500, with a best-total-loss state retained diagnostically.
History indexes the state where the loss was measured. Segmentation metrics
are never used for epoch selection. All35labels/case are retained, including
explicit absent-label exclusions, and both foreground regularity metrics use
the unchanged canonical posthoc evaluator.
