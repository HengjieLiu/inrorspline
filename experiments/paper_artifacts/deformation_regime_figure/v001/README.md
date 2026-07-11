# Deformation Regime Composite v001

This artifact builds a paper-facing composite figure for the deformation-regime
comparison between OASIS brain registration and LungCT/4DCT registration.

Primary review surface:

- `deformation_regime_composite_review.ipynb`

Generated outputs:

- `deformation_regime_composite.png`
- `deformation_regime_composite.pdf`
- `deformation_regime_composite.svg`
- `deformation_regime_composite_manifest.json`

The notebook has a top controls cell where visible figure text, font sizes,
colors, layout values, output formats, and image crop bounds can be edited.
The left panels reuse existing deformation-overlay images from
`experiments/paper_artifacts/motion_pattern_figure/v001/`; their embedded titles
and colorbars are cropped out by default, and editable labels/colorbars are drawn
by the notebook.

Default statistical comparison:

- OASIS: `OASIS SINR cps4 alpha1000`
- 4DCT: `4DCT IDIR default`

Statistical sources:

- `experiments/2026-06-25_motion_regime_seed3_first10_current_methods/tables/case_metrics_comprehensive.csv`
- `experiments/2026-06-25_motion_regime_seed3_first10_current_methods/curves/lowpass_curves.csv`

The SDlogJ panel intentionally uses positive-Jacobian foreground SDlogJ loaded
from the per-run `summary.json` files. This matches the Pareto-figure convention
and avoids the inflated OASIS values produced by the older motion-regime CSV
column that clipped nonpositive Jacobians before taking the log.

Interpretation caveat: these panels characterize estimated displacement fields
under their documented dataset/method protocols. They should support a
deformation-regime argument, not a method-only comparison across unmatched
preprocessing protocols.
