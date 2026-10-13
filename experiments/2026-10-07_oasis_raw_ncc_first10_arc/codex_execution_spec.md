# First10 raw-NCC, VFA and SITReg ARC report

User requested SITReg results and ARC-like trade-off figures with SD ellipses.
This is posthoc evaluation/plotting only: no fitting, retraining or changes to
existing saved fields or GPU jobs. Use the approved container UID:GID1009:1020.

- Same seed3 shuffled first10, fixed/moving direction and canonical absent-label
  Dice/HD95/symmetric ASD as P40/P41. Re-evaluate20 original best-Dice VFA/SITReg
  fields, audit shape/layout/units/hashes, and verify VFA against P40 plus SITReg
  rank0 against the prior canonical reference. Use the common Jacobian routine.
- Reuse40 accepted P41 raw-NCC rows at lambda30/100/300/1000. Verify case IDs,
  completion, source summaries, metric implementation and cached prefix identity.
  Do not present rank0-only lower lambdas as first10 operating points.
- Main three panels: Dice/HD95/ASD versus foreground SD(log J). Ellipses are
  axis-aligned marginal +/-1 sample SD across10 cases (ddof1), not confidence
  regions or covariance ellipses. Full width/height =2SD, matching the existing
  presentation ARC drawing convention.
- SD(log J) uses positive-J fixed-raw-image-positive voxels, after the common
  two-voxel boundary crop; within-image SD uses ddof0. Include a companion
  three-panel folding-percentage trade-off and a folding table to avoid hiding
  nonpositive Jacobians behind positive-only log statistics.
- Export PNG/SVG/PDF and casewise/mean-SD CSV/JSON/Markdown to a distinct external
  review directory `oasis_seed3_first10_raw_ncc_vfa_sitreg_arc_v1`.
- Unit-test sample SD, ellipse geometry and incomplete-cohort rejection. Smoke
  rank0 evaluation before all20 baseline evaluations; visually inspect final PNG.
- All first10 raw-NCC tuning remains exploratory. Ranks1–9 summaries are
  descriptive only, not held-out confirmation for the extended lambda sweep.
