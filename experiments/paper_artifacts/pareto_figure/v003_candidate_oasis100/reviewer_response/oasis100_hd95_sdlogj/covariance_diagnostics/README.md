# Covariance Diagnostics

These outputs use paired case-level OASIS100 HD95 and foreground SDlogJ values for each method/alpha group.

## Files

- `case_pairs.csv`: paired case-level HD95/SDlogJ rows.
- `covariance_summary.csv`: covariance, Pearson correlation, Spearman correlation, and rotated ellipse parameters by group.
- `covariance_rotated_ellipses__hd95_mean_vs_log_jdet_std_fg.png`: 2D Pareto-style diagnostic using rotated 1-SD covariance ellipses.
- `correlation_heatmap_by_method_alpha.png`: Pearson correlation by method and alpha.
- `manifest.json`: provenance and file checks.

## Reading Notes

- Raw covariance is unit-dependent, so Pearson `r` and Spearman `rho` are the main interpretable summaries.
- `Mean HD95`: lower is better.
- Ellipse angle is derived from the sample covariance matrix of `(SDlogJ, HD95)` across cases.
- Missing/empty output files: `0`.

## How To Read The Figures

`covariance_rotated_ellipses__hd95_mean_vs_log_jdet_std_fg.png` is the
HD95-vs-SDlogJ Pareto view with rotated covariance ellipses. Each point is a
method/alpha setting averaged over OASIS `N=100`. The ellipse shows paired
case-level spread: horizontal spread is SDlogJ variation, vertical spread is
HD95 variation, and ellipse tilt shows whether the two metrics covary across
cases. An upward left-to-right tilt means cases with higher SDlogJ tend to have higher, worse HD95.

`correlation_heatmap_by_method_alpha.png` is the compact numerical summary of
the same question. Each cell is the Pearson correlation between case-level
SDlogJ and HD95 for one method/alpha group, ranging from `-1` to `1`. In this
diagnostic package, finite correlations range from about `0.52` to `0.80`, so higher case-level SDlogJ is associated with higher, worse HD95 within
each plotted method/regularization setting. Blank cells indicate alpha values
not used by that method family.
