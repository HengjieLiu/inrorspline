# OASIS100 HD95-SDlogJ Reviewer Diagnostics

This package is a reviewer-response diagnostic beside the current Figure 2 artifact; it does not replace the paper figure.

## Links

- Reviewer comment: `docs/offgrid_submission/review.md:76`
- Current OASIS100 Pareto package: `experiments/paper_artifacts/pareto_figure/v003_candidate_oasis100/`
- Current OASIS100 Pareto source rows: `experiments/paper_artifacts/pareto_figure/v003_candidate_oasis100/oasis100_pareto_source_rows.csv`
- Covariance diagnostics: `covariance_diagnostics/README.md`
- Violin marginal diagnostics: `violin_marginals/README.md`
- Innovative visualization trials: `innovative_visualization_trials/README.md`

## Scope

- Cohort: OASIS seed-3 shuffled first100, case ranks `0-99`.
- Paired metrics: `hd95_mean` and `log_jdet_std_fg`.
- Sweep case rows: `3600`.
- Baseline case rows: `500`.
- Diagnostic groups: `41`.

## Interpretation

- `Mean HD95`: lower is better.
- Rotated covariance ellipses and correlation tables address whether case-level SDlogJ and HD95 move together.
- Violin plots disentangle the marginal spread along each axis, but they do not show covariance.
