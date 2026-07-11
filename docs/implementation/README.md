---
doc_type: implementation_index
status: active_draft
---

# Implementation Notes

This folder is the code-derived technical hub for `inrorspline`. It records how
the current code implements the method families, training objectives,
regularity metrics, and experiment-specific adaptations that matter for paper
writing.

Use these notes when converting code behavior into paper Methods text, figure
captions, protocol caveats, and implementation claims. Keep scientific protocol
definitions in `docs/03_methods.md`; keep run ledgers in
`docs/04_experiment_registry.md`; keep experiment-specific audits and plots in
`experiments/` or `docs/reports/`.

## Files

- `code_sources.md`: map from implementation claims to repo-local source files
  and experiment records.
- `idir_sinr_architecture_training.md`: IDIR and SINR architecture, training
  losses, and OASIS/LungCT protocol differences.
- `directcp_bspline_optimization.md`: DirectCP optimization, cubic B-spline
  formulas, analytic derivatives, finite-difference comparisons, and
  multistage knot insertion.
- `paper_claims_crosswalk.md`: paper claims mapped to code paths, experiment
  evidence, metrics, and caveats.

## Review Safety

This folder is intended to be safe for anonymous review. Do not add raw data,
checkpoints, dense fields, private host paths, credentials, private usernames,
private hostnames, W&B entities, or identity-breaking metadata.

Use repo-relative paths, dataset handles, protocol tags, and environment
variables when a path or artifact location is needed.
