---
doc_type: literature_note
status: active_draft
citation_key: sideri2025arc
---

# Evaluation Of Deformable Image Registration Under Alignment-Regularity Trade-Off

## Source And Code

- Paper: Sideri-Lampretsa, Rueckert, and Qiu, arXiv 2025.
- Local private PDF: `private/literature/Evaluation of Deformable Image Registration Under Alignment-Regularity Trade-Off.pdf`
- Code listed by the paper: anonymous ARC repository.
- Current repo status: no confirmed official ARC submodule.
- Related checked-in code: [external/MIR](../../external/MIR) for registration benchmarking utilities,
  but it should not be treated as the official ARC implementation unless verified.

## Core Idea

This paper argues that single-point registration comparisons are often
misleading because alignment accuracy and deformation regularity trade off
against each other. A method can appear better by accepting more folding or
larger Jacobian variation, while another method may be preferable in a
low-folding operating regime.

The proposed evaluation object is the alignment-regularity characteristic
curve. Instead of plotting performance against regularization weight, ARC plots
alignment metrics directly against measured regularity metrics. This makes
methods with different loss scales and regularization parameterizations more
comparable.

## Method Details

- Sweep regularization weights for each method.
- Measure accuracy and regularity on the test set.
- Plot accuracy versus regularity, not accuracy versus the raw lambda value.
- Alignment metrics:
  - OASIS: Dice over labels.
  - NLST: landmark TRE.
- Regularity metrics:
  - folding ratio from negative Jacobian determinants.
  - standard deviation of log-Jacobian.
- Amortized variant: use a HyperNetwork-style approach to sample regularization
  levels more densely without training a separate model at every lambda.

## Experiments And Metrics

- Datasets: OASIS brain MRI and NLST inhale/exhale lung CT.
- Methods include VoxelMorph, TransMorph, LapIRN, RCN, MIDIR, displacement-field
  variants, and SVF/diffeomorphic variants.
- The paper highlights that conclusions can change depending on where the ARC
  curve is sliced. A method with higher maximum Dice can be less attractive in a
  low-folding regime.
- The paper also argues that SVF models may be underrated when judged only at
  overly strict zero-folding operating points.

## Relevance To Inrorspline

This is a direct evaluation standard for the project. The repo's core claim is
not just "which method has higher Dice or lower TRE", but which method occupies
better alignment-regularity trade-off regions under matched protocols.

The ARC idea should shape our experiment outputs:

- Report Dice/TRE together with bending energy, diffusion energy, log-Jacobian
  spread, and folding percentage.
- Compare methods along curves or matched regularity regimes, not only at each
  method's best alignment point.
- Treat OASIS seed-3 first10 and LungCT protocol tags as preconditions for fair
  ARC-style comparisons.

## Protocol Cautions

- ARC curves do not remove the need for matched preprocessing. A curve on native
  full-volume LungCT is not directly comparable to a pTV-style cropped 1 mm
  domain curve.
- Regularity metrics can be sensitive to displacement units, image spacing,
  interpolation, and evaluation grid.
- The paper's code link is anonymous in the PDF; before depending on code,
  choose a canonical public GitHub repo and add it with `[add git external]`.

## Open Questions For This Repo

- Which regularity metric should define the primary x-axis for OASIS and LungCT:
  folding percentage, log-Jacobian spread, bending energy, or diffusion energy?
- Should the main paper show full ARC curves, low-folding zooms, or both?
- Can a small seed-3 OASIS first10 curve be used as a smoke benchmark before
  full test-set evaluation?

