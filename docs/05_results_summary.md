---
doc_type: results_summary
status: active_draft
---

# Results Summary

This repo currently carries forward audited planning reports rather than newly generated results.

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

## Source Reports

- `docs/reports/current_results_audit_and_plan.md`
- `docs/reports/reorganized_paper_points.md`
- `docs/reports/protocol_warnings/lungct_idir_ptvreg_protocol_bias.md`
- `docs/reports/protocol_warnings/ptvreg_technique_audit_directcp_multistage.md`
