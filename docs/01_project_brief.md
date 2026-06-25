---
doc_type: project_brief
status: active
---

# Project Brief

## Working Thesis

Continuous/off-grid deformation models should not be evaluated only by representation label. In deformable registration, performance depends on the transformation prior induced by parameterization, optimization, regularization, and the motion regime.

The paper should avoid broad claims like "SINR is invalid" or "IDIR fails." The stronger claim is narrower and more useful:

> In continuous medical image registration, the effective deformation prior matters at least as much as the fact that the deformation field can be evaluated off-grid.

## Main Research Questions

1. On OASIS brain registration, how much of SINR's gain remains if the spline/local-control component is optimized directly without the INR component?
2. Do alignment-regularity Pareto fronts reveal differences hidden by single operating-point Dice/TRE?
3. Why can IDIR look weak on intersubject brain registration but competitive on intrasubject LungCT registration?
4. For LungCT, how much of DirectCP performance comes from coarse-to-fine capture range rather than final control spacing alone?

## Primary Datasets

- `brain`: OASIS brain registration, with the seed-3 shuffled first10 protocol as the first paper-facing target.
- `lungct`: DIR-Lab 4DCT and related LungCT/COPD protocols, with explicit preprocessing/protocol tags.

## Main Contributions To Aim For

- A controlled OASIS Pareto comparison of IDIR, SINR, and direct B-spline control-point optimization.
- A LungCT analysis showing that motion-regime fit and coarse-to-fine optimization matter.
- A deformation-regime comparison explaining why brain and lung behave differently.
- A protocol audit that prevents unfair method-only claims when preprocessing differs.
