---
experiment_id: 2026-08-31_copd_p2e_geometry_w0p055_all10
status: completed_candidate_review
dataset: COPD10 pTV-style 1-mm landmark crop
---

# COPD P2e-geometry isoTV `0.055` candidate screen

This folder records an isolated all-ten-case COPD screen of the completed P2e
geometry and schedule with **only** the isoTV weight changed from `0.11` to
`0.055`.  It is not P2e and does not alter the frozen P2e reference or the
cross-dataset TRE report unless explicitly selected after review.

- Direction: `eBHCT -> iBHCT`.
- GPU policy: GPU `0` only; one case at a time.
- Headline endpoint: native-crop/original-domain snap-to-voxel landmark TRE in
  mm.
- Review policy: write casewise and aggregate outputs here, then await user
  review before adding any result to another report.

See [the execution specification](codex_execution_spec.md) and
[technical log](technical_log.md) for the frozen settings and acceptance gates.
The completed candidate comparison is in
[candidate_vs_p2e_review.md](candidate_vs_p2e_review.md); it has not been
copied into the cross-dataset summary report.
