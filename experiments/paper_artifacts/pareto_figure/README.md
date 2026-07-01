# Pareto Figure Artifacts

Each versioned subfolder contains standardized Pareto figures, source-row CSVs,
selected-hypervariant manifests, and LR validation reports.

Current version: `v001`.

For `v001`, OASIS uses the full configured accuracy-regularity grid and writes
both annotated review plots and paper-facing no-annotation plots. The paper
OASIS plots remove alpha labels and reference text labels such as Greedy, VFA,
and SITReg. LungCT is original snap TRE only, plotted against each configured
regularity metric with the default-exclusion design, and also writes annotated
review plots plus paper-facing no-annotation plots without ratio/pTVReg text.

Display names come from `../method_nicknames.csv`. OASIS Pareto labels keep
control spacing explicit, such as `INR-BSCP (cps2)` and
`MR-D-BSCP (8-4-2)`. LungCT Pareto labels decouple model from loss family, such
as `INR-Dense (IDIR loss)`, `MR-D-BSCP (IDIR loss)`, and
`MR-D-BSCP (pTVReg loss)`. The generated manifest records the resolved
method-ID to display-name mapping.

Typography is controlled near the top of `../build_standardized_pareto.py`.
For the current v001 regeneration, Pareto tick labels are `12 pt` and x/y axis
labels are `16 pt` for both OASIS and LungCT. OASIS legends are `10 pt`, while
LungCT legends remain `12 pt`. The
previous OASIS plots used Matplotlib-default tick and axis-label sizes
(`~10 pt`) with `6.4 pt` legends; the previous LungCT plots used `8 pt` tick
labels, `10 pt` axis labels, and `8 pt` legends. An intermediate review pass
used `16 pt` tick labels before this final `12 pt` setting.

Regularity-axis labels spell out `Foreground` instead of the shorter `FG`
abbreviation.
