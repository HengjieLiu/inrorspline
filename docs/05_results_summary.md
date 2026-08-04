---
doc_type: results_summary
status: active_draft
---

# Results Summary

This repo carries audited generated results together with planning reports.

## Strong Existing Evidence

- OASIS case000 suggests DirectCP cps4 with matched spline-style regularization can match or exceed SINR cps4, while original IDIR is weaker.
- LungCT/4DCT shows IDIR can be competitive when motion is smooth, coherent, and intrasubject.
- LungCT single-stage DirectCP cps4 fails, while multistage DirectCP improves substantially, supporting a capture-range/coarse-to-fine explanation.
- Deformation-stat reports support a brain-vs-lung regime difference: OASIS has lower displacement amplitude but more local heterogeneity; 4DCT has larger but more coherent motion.

## Not Yet Paper-Final

- Main OASIS Pareto must be rerun on seed-3 shuffled first10.
- OASIS deformation-regime stats need the same seed-3 first10 protocol.
- 4DCT matched Pareto remains incomplete.
- Baseline displacement regularity metrics need same-pipeline computation.

## Controlled DUAL DIR-Lab Baseline (2026-08-03)

- The controlled N0/N1/P0/P1 cohort completed 40/40 records on DIR-Lab 4DCT
  cases 01-10 with zero failures or retries. All runs used the same frozen DUAL
  architecture, loss, optimizer recipe, seed, and 2,500-step stage budget.
- Headline/native snap TRE (mean +/- sample SD, mm) was
  `1.0738 +/- 0.2041` for N0, `1.0649 +/- 0.2091` for N1,
  `1.0410 +/- 0.2160` for P0, and `1.0461 +/- 0.2211` for P1.
- The paired MR-shaped image-pyramid effect was small and direction-
  inconsistent: N1-N0 `-0.0089 +/- 0.0134 mm`; P1-P0
  `+0.0051 +/- 0.0147 mm`. This supports using these rows as controlled
  baselines, not claiming a robust improvement from the image schedule.
- All four controlled variants had zero nonpositive foreground Jacobian
  percentage under the shared posthoc convention. Foreground SDlogJ was
  `0.1282`, `0.1286`, `0.1135`, and `0.1138` for N0, N1, P0, and P1.
- Contextual iso-1-mm references were pTVReg `0.9191 +/- 0.1607 mm` and
  MR-D-BSCP `0.9175 +/- 0.1561 mm`. The MR-D-BSCP row uses the single
  cohort-selected `w=0.055` setting for all ten cases, never casewise minima.
- N uses native anisotropic full volumes with fixed-T00 lung-mask sampling; P
  uses a pTVReg-style isotropic-1-mm landmark crop with no lung mask in
  optimization. Only N0-N1 and P0-P1 are controlled schedule contrasts; N-P
  and external-reference comparisons retain explicit preprocessing caveats.
- Full casewise TRE, regularity, runtime, reference provenance, and acceptance
  are in
  `experiments/2026-08-02_dual_inr_dirlab_controlled_multires/camera_ready_readout.md`.

## Source Reports

- `docs/reports/current_results_audit_and_plan.md`
- `docs/reports/reorganized_paper_points.md`
- `docs/reports/protocol_warnings/lungct_idir_ptvreg_protocol_bias.md`
- `docs/reports/protocol_warnings/ptvreg_technique_audit_directcp_multistage.md`
