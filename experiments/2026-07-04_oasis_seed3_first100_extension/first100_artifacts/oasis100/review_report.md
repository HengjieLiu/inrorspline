# OASIS Seed-3 oasis100 Six-Sweep Pareto Plot Package

- Generated: `2026-07-06 14:11:42 +0000`
- Sweep rows: `3600`
- Baseline rows: `500`
- Population rows: `41`
- Styles: line segments, line segments with x/y 1-SD bars, and axis-aligned 1-SD ellipses.
- Added DirectCP `8-4-2` as a sixth full six-alpha curve at canonical `lr=1e-3`.
- Foreground log-J metric: `log_jdet_std_fg`, the foreground standard deviation of log Jacobian determinant.

## Best/Reference Points

- IDIR: best Dice alpha `1`, best HD95 alpha `10`.
- SINR cps4: best Dice alpha `1000`, best HD95 alpha `10000`.
- DirectCP cps4: best Dice alpha `1000`, best HD95 alpha `10000`.
- SINR cps2: best Dice alpha `1000`, best HD95 alpha `10000`.
- DirectCP cps2: best Dice alpha `1000`, best HD95 alpha `10000`.
- DirectCP 8-4-2: best Dice alpha `1000`, best HD95 alpha `10000`.
- BS1 VoxelMorph: Dice `0.7682`, HD95 `2.5402`, symmetric ASD `0.5603`.
- BS2 TransMorph: Dice `0.7953`, HD95 `2.1486`, symmetric ASD `0.4549`.
- BS3 VFA: Dice `0.8216`, HD95 `1.8848`, symmetric ASD `0.3709`.
- BS4 SITReg: Dice `0.8246`, HD95 `1.8157`, symmetric ASD `0.3556`.
- Greedy: Dice `0.8087`, HD95 `2.0183`, symmetric ASD `0.4112`.

## Files

- `population_summary_oasis100_with_saved_baselines.csv/json`
- `population_label_metrics_oasis100_with_saved_baselines.csv`
- `small_plots/`: individual accuracy-regularity panels.
- `big_plots/`: high-DPI contact sheets and spotlight plots.
- Generated figure files: `162`

