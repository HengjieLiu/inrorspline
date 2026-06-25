# OASIS Seed-3 First10 Alpha Sweep Technical Log

## Status

- Code prepared for review.
- No first10 training jobs launched.
- Launch remains gated on user approval.

## Known Limitations Before Launch

- This is a BE-trained sweep only. Diffusion is reported posthoc as native
  analytic diffusion and forward-difference diffusion; diffusion-regularized
  training sweeps are not included in this run.
- DirectCP cps4 uses only lr `1e-3`. If DirectCP is unstable or weak, LR remains
  a possible confound because lr `1e-2` is not included in this first main run.
- IDIR alpha values are not directly comparable to SINR/DirectCP alpha values
  because original IDIR BE keeps its public scaling convention.
- The first10 subset is small. Population ARC curves are appropriate for the
  project first target, but final claims should still be phrased as first10
  evidence unless a larger OASIS subset is run.
- `loss_history_all.csv` is expected to be large because every epoch is retained
  for 240 jobs.
- Dense displacement output is retained for every job, so large output storage
  should be budgeted at roughly `25-30 GB`.

## Acceptance Checks Planned

- Exactly `240` summaries: 10 case ranks times 4 methods times 6 alphas.
- No missing, duplicate, or unexpected `(case_rank, method, alpha, lr)` combos.
- Each rank resolves to the expected seed-3 shuffled pair.
- Every run has `DONE`, `summary.json`, `history.csv`, case metrics,
  regularity metrics, BE/diffusion audit CSVs, model state, mask audit, and
  dense displacement output.
- IDIR summaries must use `fixed_image_positive` training samples and
  coordinate-autograd native analytic BE/diffusion.
- SINR/DirectCP summaries must use no image-loss foreground mask and B-spline
  native analytic BE/diffusion.
- Population ARC curves must aggregate cases by method/alpha before plotting
  the Pareto/front curves.
