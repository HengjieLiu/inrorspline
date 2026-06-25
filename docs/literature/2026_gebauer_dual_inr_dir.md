---
doc_type: literature_note
status: active_draft
citation_key: gebauer2026dual_inr_dir
---

# Robust Multi-Scale Implicit Neural Representations For Large-Deformation Lung Registration

## Source And Code

- Paper: Gebauer, Nielsen, Madesta, Werner, and Sentker, MIDL 2026.
- Local private PDF: `private/literature/2026_DUAL-INR-DIR_Robust_Multi_Scale_Implicit_Neural_Representations_for_Large_Deformation_Lung_Registration.pdf`
- Official code listed by the paper: `https://github.com/IPMI-ICNS-UKE/DUAL-INR-DIR`
- Current repo status: official code is not yet added as an `external/`
  submodule.
- Related code: [external/IDIR](../../external/IDIR), [external/pTVreg](../../external/pTVreg)

## Core Idea

This paper extends SIREN-based INR registration for large thoracic deformation.
The main claim is that a standard single-branch SIREN can fit fine local motion
but may fail to capture large global motion robustly. The proposed solution is a
dual-branch INR trained with a classical coarse-to-fine schedule.

The model decomposes motion into a coarse low-frequency branch and a fine
high-frequency branch. Gaussian-smoothed images guide the coarse branch first;
the fine branch is activated later to refine local details.

## Method Details

- Transformation model: residual INR deformation `phi(x) = x + u(x)`.
- Architecture: two SIREN MLP branches, coarse and fine, each with two hidden
  layers of 256 units in the reported setup.
- Frequency bias:
  - coarse branch uses lower SIREN frequency, reported as `omega = 10`.
  - fine branch uses higher SIREN frequency, reported as `omega = 30`.
- Schedule:
  - Phase 1 optimizes only the coarse branch on heavily smoothed images.
  - Phase 2 reduces smoothing and activates the fine branch with a smaller
    initial scale.
- Similarity: normalized cross-correlation.
- Sampling: random points from inside the fixed lung mask.
- Regularization: curvature, negative-Jacobian penalty, and lobe-mask semantic
  loss.
- Masks: lung and lung-lobe masks are generated with TotalSegmentator in the
  paper setup.

## Experiments And Metrics

- Datasets: DIR-Lab 4DCT and DIR-Lab COPDgene, 10 cases each.
- Primary metric: landmark TRE in millimeters.
- Additional metrics: lobe Dice and negative-Jacobian percentage.
- Reported averages:
  - 4DCT: proposed method about `0.98 mm` mean TRE.
  - COPDgene: proposed method about `1.23 mm` mean TRE.
- The paper reports that isoPTV remains stronger on average, but the dual-branch
  INR substantially improves robustness over the official/default IDIR setup on
  COPDgene, where standard IDIR can fail badly on one case.

## Relevance To Inrorspline

This is a direct LungCT/large-deformation INR reference and is highly relevant
to this repo's deformation-regime question. It suggests that "INR underperforms
on LungCT" may be too broad: failure may depend on optimization path, motion
scale separation, and regularization rather than the INR representation alone.

For our study, it motivates these controls:

- single-stage IDIR versus multi-stage/coarse-to-fine IDIR.
- single-branch INR versus dual/multi-branch or frequency-decoupled INR.
- DirectCP multistage schedules versus INR multistage schedules.
- pTV/isoTV as a strong classical large-deformation baseline.

## Protocol Cautions

- The paper uses generated lung and lung-lobe masks. That is not a clean
  comparison to pTVreg configurations that do not optimize with lung masks.
- TRE uses a snap-to-voxel convention; this must be aligned before comparing
  against repo outputs.
- TotalSegmentator lobe masks introduce a semantic supervision/regularization
  source that should be tagged separately from pure intensity registration.
- For repo LungCT work, do not compare this directly with native IDIR, pTVreg,
  SINR, or DirectCP unless preprocessing, masks, evaluation convention, and
  regularization are explicitly tagged.

## Open Questions For This Repo

- Add the official DUAL-INR-DIR code as a submodule?
- Should a minimal dual-branch/coarse-to-fine INR be reproduced as a method
  family, or only discussed as literature?
- Can DirectCP's multistage schedule explain much of the same robustness without
  adding INR branch complexity?

