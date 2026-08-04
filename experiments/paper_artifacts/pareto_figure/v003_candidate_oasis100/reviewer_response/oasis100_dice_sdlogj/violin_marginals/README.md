# Violin Marginal Diagnostics

These outputs split the original 2D Dice-SDlogJ relationship into separate 1D marginal distributions.

## Files

- `violin_source_rows.csv`: paired case-level rows used for both violin plots.
- `violin_marginals_full_alpha_grid.png`: Dice and SDlogJ violins for the full alpha grid plus baselines.
- `violin_marginals_selected_points.png`: compact view for best/reference sweep points plus baselines.
- `manifest.json`: provenance and file checks.

## Reading Notes

- Violin plots reduce overlap and make each axis spread easier to inspect.
- Because the axes are separated, these plots do not show covariance; use the covariance diagnostics for that question.
- Missing/empty output files: `0`.
