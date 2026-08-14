# Schedule-Matched Population Diagnostics v001

This versioned artifact contains supplementary/reviewer diagnostics. It is not
the final compact main-paper figure.

Primary review files:

- `oasis_population_diagnostics_4x5.png`
- `dirlab_population_diagnostics_4x5.png`
- `diagnostic_comparison.md`
- `validation_report.md`
- `visual_review.md`
- `figure_manifest.json`

The historical population index remains unchanged. The local
`camera_ready_source_index.csv` preserves 150 records and replaces only the ten
DIR-LAB Panel A MR-D-BSCP rows with the accepted `400/400/800/900` refresh.

Regenerate inside the declared project container with portable roots:

```bash
python scripts/experiments/fit_modes_population/render_schedule_matched_diagnostics.py \
  --population-run-root "${POPULATION_RUN_ROOT}" \
  --refresh-run-root "${DIRLAB_REFRESH_RUN_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --output-root experiments/paper_artifacts/fit_modes_population_diagnostics/v001_schedule_matched
```

The renderer is non-destructive and refuses to overwrite an existing versioned
output directory. Use a new version directory for later revisions.
