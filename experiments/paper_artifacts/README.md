# Final-Paper Artifact Standards

This folder is the source of truth for final-paper figure and table display
names, hyperparameter-variant selection, and versioned review artifacts.

## Files

- `artifact_standard.json`: canonical LR policy, artifact versions, selected
  hypervariants, temporary exceptions, and TODOs.
- `method_nicknames.csv`: editable method display-name map. Edit the
  `nickname` column only when changing paper-facing method names.
- `best_result_table/v001/`: standardized best-result table outputs.
- `pareto_figure/v001/`: standardized Pareto figure outputs.
- `fitting_figure/v001/`: standardized field-fit rRMSE figure outputs.

## Current LR Standard

- IDIR: `1e-4`
- SINR: `1e-4`
- DirectCP: `1e-3`
- DirectCP multistage: `1e-3`

The current OASIS DirectCP 8-4-2 row remains a documented v001 exception at
`lr=1e-2` because the canonical `lr=1e-3` rerun is not yet available. Replace
that row after the rerun is complete.

## Nickname Workflow

Generators should read `method_nicknames.csv` and use `nickname` for labels.
The default nickname is the current original name, so the first standardized
rerun is label-preserving. Future paper-name changes should happen in this CSV
instead of being hard-coded in plotting or table scripts.

For final paper Pareto figures and best-result tables, OASIS labels emphasize
the model family plus B-spline control spacing, for example
`INR-BSCP (cps2)`, `D-BSCP (cps4)`, and `MR-D-BSCP (8-4-2)`. LungCT labels
separate parameterization/model from optimization-loss family. In those labels,
`IDIR loss` is the paper-facing shorthand for the IDIR-style/global-NCC+BE
family used by those runs, and `pTVReg loss` denotes the pTVReg-style
LCC+isoTV objective family or pTVReg reference.

## Pareto Figure Metric Scope

- OASIS Pareto artifacts keep the full configured accuracy-regularity grid.
- OASIS Pareto `v001` writes two plot trees: an annotated review tree with
  alpha/reference-point labels, and a paper-facing no-annotation tree that
  removes alpha labels and reference text labels such as Greedy, VFA, and
  SITReg because those methods are already identified in the legend.
- LungCT Pareto artifacts use original snap TRE as the only accuracy metric,
  plotted against each configured regularity metric. Do not generate LungCT
  original-subvoxel, reg1mm-snap, or reg1mm-subvoxel Pareto panels for the
  paper-facing `v001` package.
- LungCT Pareto `v001` also writes annotated review plots and paper-facing
  no-annotation plots. The no-annotation plots remove ratio labels and pTVReg
  point text because the plotted methods are already identified in the legend.
