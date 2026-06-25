---
doc_type: literature_review
status: active_draft
---

# Literature Review

## Anchor Papers

The starting kit references these local literature anchors from the previous workspace:

- IDIR: `2022_IDIR_Implicit_Neural_Representations_for_Deformable_Image_Registration.pdf`
- SINR: `2024_SINR_Spline_enhanced_Implicit_Neural_Representation_for_Multimodal_Registration.pdf`
- ARC/Pareto trade-off evaluation: `Evaluation of Deformable Image Registration Under Alignment-Regularity Trade-Off.pdf`

Structured reading notes now live under `docs/literature/`:

- `docs/literature/2022_wolterink_idir.md`
- `docs/literature/2024_sideri_sinr.md`
- `docs/literature/2026_gebauer_dual_inr_dir.md`
- `docs/literature/2025_sideri_arc.md`

Code links are tracked as Git submodules under `external/` where available:

- [external/IDIR](../external/IDIR): IDIR official code.
- [external/SINR](../external/SINR): SINR official code.
- [external/pTVreg](../external/pTVreg): pTV/isoTV baseline code for LungCT protocol analysis.
- [external/MIR](../external/MIR): related registration benchmarking utilities; not yet verified
  as the official ARC code release.

## TODO

- Move bibliographic metadata into a review-safe form.
- Replace private absolute paths with either checked-in citation metadata or ignored private path notes.
- Add related B-spline, pTVReg, Greedy, VoxelMorph/learning-based, and deformation regularity evaluation references.
- Add the official DUAL-INR-DIR code as an external submodule if we decide to
  reproduce or audit that method directly.
