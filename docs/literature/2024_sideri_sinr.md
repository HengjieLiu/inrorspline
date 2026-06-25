---
doc_type: literature_note
status: active_draft
citation_key: sideri2024sinr
---

# SINR: Spline-Enhanced Implicit Neural Representation For Multi-Modal Registration

## Source And Code

- Paper: Sideri-Lampretsa et al., MIDL 2024.
- Local private PDF: `private/literature/2024_SINR_Spline_enhanced_Implicit_Neural_Representation_for_Multimodal_Registration.pdf`
- Official code: [external/SINR](../../external/SINR)
- Dense-INR comparator: [external/IDIR](../../external/IDIR)

## Core Idea

SINR combines a coordinate INR with a B-spline free-form deformation model. The
INR is not asked to output a dense displacement at every voxel. Instead, it
predicts displacements at sparse FFD control-point coordinates, and the B-spline
FFD interpolation produces the dense deformation.

This makes control-point spacing part of the deformation prior. Larger spacing
constrains the transformation toward smoother/global motion; smaller spacing
permits more local motion. The paper argues this implicit smoothness reduces the
folding issues seen in dense INR registration while preserving pairwise
optimization and continuous coordinate modeling.

## Method Details

- Transformation model: B-spline FFD deformation with INR-predicted control
  point displacements.
- Network activations: ReLU and SIREN variants are evaluated.
- Similarity:
  - Mono-modal experiments use normalized cross-correlation.
  - Multi-modal experiments use differentiable normalized mutual information.
- Regularization: bending energy, as in classical FFD registration.
- Control spacing: best reported CamCAN setting uses `2 mm` control-point
  spacing.
- Hyperparameters include SIREN frequency `omega`, regularization weight, and
  folding-based early stopping.

## Experiments And Metrics

- Dataset: CamCAN inter-subject brain MRI.
- Tasks: T1w-to-T1w and T1w-to-T2w registration.
- Accuracy metric: Dice over anatomical segmentations.
- Regularity metric: folding percentage via negative Jacobian determinant.
- Baselines: MIRTK, VoxelMorph, MIDIR, IDIR-ReLU, IDIR-SIREN.
- Main reported result: SINR-SIREN has the highest Dice in both mono-modal and
  multi-modal settings while reducing folding compared with dense SIREN-IDIR.

## Relevance To Inrorspline

SINR is the central hybrid baseline for this repo because it mixes INR fitting
with spline/local-control structure. It motivates the main causal question:

```text
Are gains due to the INR, the spline/FFD representation, the optimization path,
regularization, or deformation-regime fit?
```

This paper makes DirectCP an important control: if DirectCP performs similarly
to SINR under matched FFD spacing, loss, and regularization, then the INR may be
less important than the local-control spline prior.

## Protocol Cautions

- SINR paper experiments are brain MRI, not LungCT.
- SINR's multi-modal result depends on efficient NMI over sparse control-point
  sampling; this is not automatically comparable to dense voxel sampling.
- Folding thresholding is used during hyperparameter choice, so comparisons must
  report both accuracy and regularity.
- For this repo's OASIS work, use the seed-3 shuffled protocol in
  `docs/standards/oasis_brainmr_protocol.md`.

## Open Questions For This Repo

- Which control spacings (`cps4`, `cps2`, etc.) correspond most fairly to SINR's
  reported spacing/regularity regime?
- Does SINR's advantage remain when DirectCP uses the same FFD basis and matched
  optimizer schedule?
- How should NMI/NCC choices be aligned across mono-modal brain, multimodal
  brain, and LungCT experiments?

