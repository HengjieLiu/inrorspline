# Publication Best-Result Tables

Publication-formatted tables derived from `best_result_table/v002_candidate`.
Values are formatted as `mean (std)` using the selected-row raw metrics.

## Files

- `oasis20_publication_main.md`, `oasis20_publication_main.csv`, `oasis20_publication_main.tex`
- `oasis20_publication_extended.md`, `oasis20_publication_extended.csv`, `oasis20_publication_extended.tex`
- `oasis20_publication_main_with_references.md`, `oasis20_publication_main_with_references.csv`, `oasis20_publication_main_with_references.tex`
- `lungct10_publication.md`, `lungct10_publication.csv`, `lungct10_publication.tex`
- `lungct10_publication_multistage_focus.md`, `lungct10_publication_multistage_focus.csv`, `lungct10_publication_multistage_focus.tex`

## Notes

- OASIS rows use the seed-3 first20 selected rows from the v002 candidate table.
- LungCT `TRE (mm)` is the original/native-crop snap TRE column.
- LungCT rows remain protocol-tagged comparison rows; do not interpret them as clean method-only comparisons unless preprocessing, metric, mask, and loss conventions are matched.

## Rebuild

```bash
python experiments/paper_artifacts/best_result_table/build_publication_tables.py
```
