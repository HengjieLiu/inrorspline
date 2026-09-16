# Best Result Table Artifacts

Generated final-paper best-result tables, selected-row manifests, and LR
validation reports are external-only. Their logical home is
`${ARTIFACT_ROOT}/repo_experiments/paper_artifacts/best_result_table/<version>/`;
this repository retains the builder and table-format specification only.

Automatic-selection versions: `v001`, `v002_candidate`.

Publication-formatted tables: `v002_publication`.

Rebuild the publication tables with:

```bash
python experiments/paper_artifacts/best_result_table/build_publication_tables.py \
  --source-dir "$ARTIFACT_ROOT/repo_experiments/<migrated-selected-row-directory>"
```

`--source-dir` is required and must be an absolute migrated generated-artifact
directory below `${ARTIFACT_ROOT}/repo_experiments/`. The publication output
defaults to
`${ARTIFACT_ROOT}/repo_experiments/paper_artifacts/best_result_table/v002_publication`.
An explicit `--output-dir` must remain below the same external review store.

Display names come from `../method_nicknames.csv`. OASIS rows use compact model
names with explicit control spacing, for example `INR-BSCP (cps4)` and
`D-BSCP (cps2)`. LungCT rows separate the parameterization/model from the
optimization loss family, for example `INR-Dense (IDIR loss)` and
`MR-D-BSCP (pTVReg loss)`. Here `IDIR loss` is the paper-facing shorthand for
the IDIR-style/global-NCC+BE family, while `pTVReg loss` denotes the pTVReg-style
LCC+isoTV objective family or pTVReg reference. The table manifest records the
resolved method-ID to display-name mapping.
