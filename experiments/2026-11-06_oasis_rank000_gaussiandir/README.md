# P83: GaussianDIR on OASIS rank 0

Status: four frozen v1 pilot endpoints and the paper-loss convergence/resource
follow-up are complete. Both extended trajectories remain explicitly unresolved
at their 20-minute operational ceiling.
Date: 2026-11-06, America/Los_Angeles.

Approved follow-up: [paper-loss convergence, speed and GPU memory](convergence_spec.md),
with [frozen settings](protocol_convergence_v1.json). The
[LNCC9²/diffusion prototype](lncc_diffusion_spec.md) is deferred after two smokes;
no full standardized-loss runs were launched.

- [Technical report: paper, code and discrepancies](../../docs/implementation/gaussiandir_paper_code_audit.md)
- [Execution specification and decisions](codex_execution_spec.md)
- [Concrete pilot configuration](protocol_v1.json)
- [Technical log and verification limits](technical_log.md)
- [Runtime pins and repeatable commands](runtime.md)
- [Output-free review notebook](review.ipynb)
- [Output-free convergence/resource notebook](convergence_review.ipynb)
- [Reproducible synthetic audit source](../../scripts/experiments/gaussiandir/audit_upstream.py)
- [Pinned upstream source](../../external/GaussianDIR/README.md)

The two intended arms use fixed `0333`, moving `0115`, the first seed-3 shuffled
OASIS test pair:

1. Supervised reconstruction of the original saved SITReg lambda-1 displacement
   field, matching the target used by P81/P82.
2. Independent image-only GaussianDIR registration from identity, compared with
   the same saved SITReg reference using the complete shared accuracy/regularity
   metric suite.

The released defaults differ substantially from the full published model. The
report separates a released isotropic/translation control from the selected
corrected anisotropic/rigid/multiscale reconstruction. The user selected frozen
coarse then fine residual stages with literal paper caps, and full-box supervised
MSE with fixed populations. Native validation precedes the four pilot endpoints.

All four endpoints completed2,000 steps. Full-box fitting RMSE is0.3721mm for
the released control and0.2402mm for the corrected model. Independent image
registration reaches Dice0.7496/0.7676 and foreground folds4.0923%/1.7452%,
versus SITReg0.7987 and0.000697%. This single-case pilot does not match SITReg
accuracy/folding; the two GaussianDIR variants also differ in capacity and loss.
The [curated results summary](../../docs/05_results_summary.md) gives the main
numbers. All metrics, per-label rows, figures, report and executed review notebook
are external under
`${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_v1/pilot_20261106_v1/review_v2/`.

## Paper-loss convergence and speed follow-up

The additional study retained the released control and corrected reconstruction,
used sampled signed NCC + TV at journal lambda 8, and measured fresh 200-, 500-
and 2,000-step schedules three times per variant. Both convergence trajectories
used the full 1,200-second allowance; the control selected step 110,300 (Dice
0.767119) and the corrected model selected step 31,400 (Dice 0.785612), but
neither met the plateau rule. See [the frozen protocol](convergence_spec.md) and
the external report at
`${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_convergence_v1/execution_20261106_v1/report_v1/report.md`.
The executed notebook, all35-label metrics, resource traces and tables are next
to that report. Posthoc concurrent NVML PID attribution is limited by the
container namespace; registration process peaks and per-evaluation Torch peaks
are recorded.

All generated outputs belong under `${ARTIFACT_ROOT}`. Run key:
`oasis_rank000_gaussiandir_v1`, execution `pilot_20261106_v1`; use a new execution
ID for each frozen protocol.
The existing external synthetic receipt is
`${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_gaussiandir_v1/audit_20261106/synthetic_audit.json`.
This folder contains hand-authored source documentation only.
