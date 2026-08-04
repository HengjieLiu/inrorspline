# Covariance Diagnostics

These outputs use paired case-level OASIS100 Dice and foreground SDlogJ values for each method/alpha group.

## Files

- `case_pairs.csv`: paired case-level Dice/SDlogJ rows.
- `covariance_summary.csv`: covariance, Pearson correlation, Spearman correlation, and rotated ellipse parameters by group.
- `covariance_rotated_ellipses__dice_mean_vs_log_jdet_std_fg.png`: 2D Pareto-style diagnostic using rotated 1-SD covariance ellipses.
- `correlation_heatmap_by_method_alpha.png`: Pearson correlation by method and alpha.
- `manifest.json`: provenance and file checks.

## Reading Notes

- Raw covariance is unit-dependent, so Pearson `r` and Spearman `rho` are the main interpretable summaries.
- Ellipse angle is derived from the sample covariance matrix of `(SDlogJ, Dice)` across cases.
- Missing/empty output files: `0`.

## How To Read The Figures

`covariance_rotated_ellipses__dice_mean_vs_log_jdet_std_fg.png` is the
Dice-vs-SDlogJ Pareto view with rotated covariance ellipses. Each point is a
method/alpha setting averaged over OASIS `N=100`. The ellipse shows paired
case-level spread: horizontal spread is SDlogJ variation, vertical spread is
Dice variation, and ellipse tilt shows whether the two metrics covary across
cases. A downward left-to-right tilt means cases with higher SDlogJ tend to
have lower Dice.

`correlation_heatmap_by_method_alpha.png` is the compact numerical summary of
the same question. Each cell is the Pearson correlation between case-level
SDlogJ and Dice for one method/alpha group, ranging from `-1` to `1`. In this
diagnostic package, the correlations are consistently negative, about `-0.82`
to `-0.46`, so higher case-level SDlogJ is associated with lower Dice within
each plotted method/regularization setting. Blank cells indicate alpha values
not used by that method family.
