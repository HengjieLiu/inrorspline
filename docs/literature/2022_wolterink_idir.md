---
doc_type: literature_note
status: active_draft
citation_key: wolterink2022idir
---

# Implicit Neural Representations for Deformable Image Registration

## Source And Code

- Paper: Wolterink, Zwienenberg, and Brune, MIDL 2022.
- Local private PDF: `private/literature/2022_IDIR_Implicit_Neural_Representations_for_Deformable_Image_Registration.pdf`
- Official code: [external/IDIR](../../external/IDIR)
- Related LungCT baseline code: [external/pTVreg](../../external/pTVreg)

## Core Idea

IDIR treats pairwise deformable registration as instance-specific optimization of
a coordinate MLP. The network maps a spatial coordinate to a residual deformation
so that the warped moving image aligns with the fixed image. The method is not a
cohort-trained CNN; it optimizes one lightweight INR per image pair.

The important design choice is the SIREN-style sinusoidal activation. Compared
with ReLU MLPs, SIRENs can represent higher-frequency deformation structure and
support higher-order analytic derivatives through automatic differentiation.
That makes Jacobian, hyperelastic, and bending-energy style regularizers natural
to compute at arbitrary sampled coordinates.

## Method Details

- Transformation model: continuous residual coordinate map `Phi(x) = x + u(x)`.
- Network: MLP with ReLU or sinusoidal/SIREN activations; best reported setting
  uses SIREN with bending regularization.
- Similarity: normalized cross-correlation on sampled points.
- Sampling: points are sampled inside an inspiration/T00 lung mask for DIR-Lab
  4DCT.
- Optimization: one model per pair, 2500 epochs, Adam, learning rate `1e-4`,
  10k sampled points per epoch in the reported DIR-Lab setup.
- Regularity terms: Jacobian determinant penalty, hyperelastic penalty, and
  bending energy, all enabled by analytic derivatives of the INR output.

## Experiments And Metrics

- Dataset: DIR-Lab 4DCT, ten inhale/exhale chest CT pairs.
- Metric: target registration error over 300 landmarks per case.
- Main comparison: IDIR versus pTV/isoTV, CNN-style methods, and displacement
  before registration.
- Reported average TRE: IDIR about `1.07 mm`, compared with isoPTV about
  `0.95 mm` in the paper table.
- The paper reports no folding in the lung mask for the listed IDIR results.

## Relevance To Inrorspline

IDIR is the clean INR baseline: dense coordinate-to-displacement modeling,
instance optimization, SIREN activations, and explicit derivative-based
regularization. It is essential for separating "INR representation" from
"spline/local-control structure".

For our study, IDIR should not be treated as equivalent to SINR or DirectCP:

- IDIR predicts dense displacements through the INR.
- SINR predicts sparse/control-point displacement values and uses FFD to produce
  the dense field.
- DirectCP removes the INR and directly optimizes spline/control parameters.

## Protocol Cautions

- The IDIR paper uses lung-mask sampling, while pTVreg reports strong DIR-Lab
  numbers without lung-mask optimization. This is a protocol difference, not a
  pure method difference.
- Reported pTV/isoTV rows are sensitive to crop, spacing, interpolation, LCC
  approximation, control spacing, and snap-to-voxel convention.
- For LungCT comparisons in this repo, read
  `docs/standards/lungct_protocol_tags.md` and
  `docs/reports/protocol_warnings/` before interpreting any table.

## Open Questions For This Repo

- Which exact IDIR regularizer and alpha grid should be used for the fair Pareto
  curve?
- Should the IDIR loader use native full volume plus lung mask, or a matched
  pTV-style crop domain?
- How should IDIR regularity be normalized against spline/control-point methods?

