---
experiment_id: learn2breath_copd10_pairlocal_mask_audit
status: diagnostic_not_optimization
date: 2026-08-30
dataset: learn2breath_challenge_like_copd10_pairlocal
---

# Learn2Breath COPD10 pair-local mask audit

This diagnostic experiment audits three supplied segmentation sources on the
Learn2Breath challenge-like COPD10 pair-local images. It neither generates
masks nor performs registration or optimization.

The primary review surface is [mask_audit.ipynb](mask_audit.ipynb). It renders,
for every case, a three-row by six-column coronal overlay:

| Rows | Columns |
| --- | --- |
| R231 whole lung; TotalSegmentator full-`total` lobes; TotalSegmentator ROI-subset lobes | shared R231-derived 25%, 50%, and 75% coronal positions, each shown as adjacent INSP/fixed and EXP/moving panels |

The pair-local images share an RAS-like 1.5 mm grid. Phase `1` is
INSP/fixed and phase `2` is EXP/moving. For each case, the notebook derives one
shared posterior-to-anterior R231 extent from the union of the two phase masks;
this keeps the INSP/EXP panels at an identical coronal coordinate. It records
the posterior boundary, anterior boundary, and the displayed 25%/50%/75%
indices in the inline audit table.

Run the notebook in `py310sitreg_v1_test1` with the two required environment
variables described in [codex_execution_spec.md](codex_execution_spec.md). It
intentionally calls no figure-saving function; outputs remain inline in the
executed notebook only.

Host-specific source and mask paths are stored in the ignored private note
`private/learn2breath_copd10_pairlocal_mask_audit_paths.md`.
