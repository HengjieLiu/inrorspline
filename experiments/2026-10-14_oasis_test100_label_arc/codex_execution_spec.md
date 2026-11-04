# OASIS test100 label-wise ARC, release v001

User approved the audited defaults on 2026-10-14. Aggregation and plotting only:
no registration, inference, field loading, warping, or metric reevaluation.

- Main: NCC9/diffusion MR-D-BSCP 8→4→2→dense at 0.1/0.3/1/3;
  VFA, SITReg, FeDDA-1MF at 0.1, exact 1/3, 1, 3, 10 (19 settings).
- Supplement: four saved NCC9/diffusion 8→4→2 prefixes, each with the same
  lambda at every prefix stage. Compare with dense and original DL sweeps.
- Cohort: seed-3 shuffled ranks0–99, moving→fixed; preserve rank4.
- Y: canonical saved label Dice, HD95 and pooled-area symmetric ASD.
- X: saved whole fixed-raw-image-positive foreground SD(log J), positive J,
  canonical crop2. This is not label-local regularity. Folding remains a table
  metric. Across-case sample SD uses ddof1.
- For each label, intersect valid case ranks across all23 configurations and
  use that same subset for accuracy AND regularity. Preserve excluded rows.
  Current expectation: Right-Vessel label34 n98 (ranks35/58 excluded); others100.
- Resolve names from the official neurite-oasis seg35_labels.txt; no inferred
  FreeSurfer relabeling. Historical DL lambda1 uses distinct markers/segments.
- Main and supplementary: 35 label-specific 2x3 PNG/SVG pages, combined PDFs,
  three 7x5 overview atlases per comparison, descriptive trend/paired tables.
  Lines follow lambda order, without interpolation, smoothing or extrapolation.
  Ellipses are marginal +/-1 sample SD, NOT confidence/covariance regions.
- Stable external root: `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_test100_label_arc/`;
  append-only registry snapshots and `releases/v001/`. Never overwrite releases.
  Source pointers/hashes, rather than copies of fields, support future additions.
- GNCC, GNCC-squared, older BE/INR methods, and first10-only studies excluded
  from this release. No additional experiments authorized by this approval.

Acceptance: 23x100 case records, 23x100x35 label slots, unique rank/pair/label
keys, completion markers and field existence, source snapshot hashes unchanged,
finite canonical case metrics, label-to-case reconstruction, main19 case-level
means/SD matching accepted ARC, explicit exclusions, plotting smoke and tests.
Field bytes are not rehashed or evaluated; preserve their existing recorded
digests and check existence/size/mtime. Report that distinction explicitly.

Use the existing artifact-store container, UID:GID1009:1020, CPU only. Preserve
all unrelated worktree changes and accepted historical artifacts. No commit.
