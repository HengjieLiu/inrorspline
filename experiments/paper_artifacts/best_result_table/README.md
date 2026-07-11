# Best Result Table Artifacts

Each versioned subfolder contains generated final-paper best-result tables,
selected-row manifests, and LR validation reports.

Automatic-selection versions: `v001`, `v002_candidate`.

Publication-formatted tables: `v002_publication`.

Rebuild the publication tables with:

```bash
python experiments/paper_artifacts/best_result_table/build_publication_tables.py
```

Display names come from `../method_nicknames.csv`. OASIS rows use compact model
names with explicit control spacing, for example `INR-BSCP (cps4)` and
`D-BSCP (cps2)`. LungCT rows separate the parameterization/model from the
optimization loss family, for example `INR-Dense (IDIR loss)` and
`MR-D-BSCP (pTVReg loss)`. Here `IDIR loss` is the paper-facing shorthand for
the IDIR-style/global-NCC+BE family, while `pTVReg loss` denotes the pTVReg-style
LCC+isoTV objective family or pTVReg reference. The table manifest records the
resolved method-ID to display-name mapping.
