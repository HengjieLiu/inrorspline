# v001 vs v002 Candidate Paper Artifact Comparison

- Generated UTC: `2026-07-01T14:30:07+00:00`
- v001 table: `experiments/paper_artifacts/best_result_table/v001`
- v002 candidate table: `experiments/paper_artifacts/best_result_table/v002_candidate`
- v001 Pareto: `experiments/paper_artifacts/pareto_figure/v001`
- v002 candidate Pareto: `experiments/paper_artifacts/pareto_figure/v002_candidate`
- Recommendation: `candidate_is_standardized_review_metrics_before_promotion`

## Validation Status Counts

- Table v001: `{'matches_standard': 10, 'temporary_exception': 1, 'reference': 4}`
- Table v002 candidate: `{'matches_standard': 11, 'reference': 4}`
- Pareto v001: `{'matches_standard': 10, 'temporary_exception': 1}`
- Pareto v002 candidate: `{'matches_standard': 11}`

## OASIS DirectCP 8-4-2 Table Row

- v001 selected: `alpha=1000`
- v002 candidate selected: `alpha=1000`
- v001 Dice: `0.808 +/- 0.039`
- v002 candidate Dice: `0.809 +/- 0.038`

## Files

- `best_result_table_delta.csv`
- `oasis_directcp842_alpha_curve_delta.csv`
- `validation_status_comparison.csv`

## Side-by-Side Plots

- `experiments/paper_artifacts/comparison/v001_vs_v002_candidate/side_by_side_dice_vs_foreground_logj.png`
- `experiments/paper_artifacts/comparison/v001_vs_v002_candidate/side_by_side_hd95_vs_foreground_logj.png`
- `experiments/paper_artifacts/comparison/v001_vs_v002_candidate/side_by_side_contact_sheet_line_segments.png`
