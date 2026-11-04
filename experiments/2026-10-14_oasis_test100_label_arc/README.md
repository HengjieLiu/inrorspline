# OASIS test100 label ARC source package

Entry point: `scripts/reports/oasis_test100_label_arc.py`.
Tests: `tests/test_oasis_test100_label_arc.py`.

Run in the artifact-store container with the official `seg35_labels.txt`:

```bash
python -B scripts/reports/oasis_test100_label_arc.py \
  --artifact-root "$ARTIFACT_ROOT" --label-map "$OASIS_SEG35_LABEL_MAP" \
  --release v001
```

Run a full-cohort scalar-validation/limited-render smoke with
`--release smoke_vNNN --smoke`. This still validates all23 configurations;
only individual label pages1/34 are rendered. A smoke is not a full release.

## Adding future sweeps

1. Obtain approval for the additional methods/comparison scope. This script
   does not launch training, registration, inference or evaluation.
2. Require saved100-case predictions, canonical35-label accuracy and canonical
   foreground regularity; retain exact pair/field/checkpoint/protocol identity.
3. Extend the isolated `load_sources` adapter with a distinct configuration ID
   containing method/representation/loss/regularizer/stage weights/seed/checkpoint
   identity as appropriate. Different loss families must not silently become
   one lambda curve. Update expected configuration/coverage gates and tests.
4. Preserve official label IDs and canonical missing-label handling. Recompute
   shared valid-case support from scalar tables. If support changes, disclose
   it; older versions remain immutable and need not have identical cohorts.
5. Run tests and a new smoke, then generate a NEW release (v002, v003, ...).
   Existing releases and registry snapshots refuse overwrite. Never copy or
   modify source fields to make the new report.
6. Check plots, source hashes, case-to-label reconstruction and acceptance;
   update source-only registry/spec/log, without committing generated artifacts.

Artifact layout: `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_test100_label_arc/`
contains `registry/<release>.json` plus `releases/<release>/` with manifests,
audit records, CSV/JSON tables, main/supplementary label pages, PDF books,
overview atlases, report and acceptance hashes. Each snapshot records source
script hash and Git revision/dirty state, not just a mutable path.

The SD ellipses are marginal sample-SD displays, not covariance ellipses or
confidence intervals. All accuracy/regularity relationships are descriptive;
best per-label lambdas do not define a single deployable field.
