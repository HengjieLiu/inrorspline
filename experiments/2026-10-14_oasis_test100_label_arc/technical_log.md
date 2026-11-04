# Technical log

## Limitations

- Whole-foreground regularity is paired with label accuracy, not measured
  inside each structure. No metric reevaluation is authorized.
- Historical DL lambda1 checkpoints differ from newly trained sweep points.
- Test-set curves and best-lambda summaries are descriptive/exploratory;
  no per-label tuned field is constructed, no significance claims are made.
- SD ellipses show across-case marginal spread, not confidence or covariance.
- Existing source field digests are retained, not freshly recomputed.

## 2026-10-14

- User approved primary four-method plots, supplementary prefix comparison,
  shared foreground SD(log J), per-label2x3 figures and versioned external root.
- Read-only inventory found all required predictions and saved label metrics.
- Official seg35 metadata identifies label34 as Right-Vessel. Ranks35/58 have
  absent moving/fixed labels, respectively; use common finite subsets.
- Original VFA/SITReg legacy case accuracy exactly matches current references.
  Legacy mount aliases are recorded in provenance, without modifying sources.
- Ten unit tests passed: scalar reconstruction, duplicate/missing slots,
  pair/direction guards, shared support on both axes, NaN/infinity distinction,
  official label-map validation, source changes and lambda-order plotting.
- Full-cohort smoke accepted in separate `releases/smoke_v001/`; main and
  supplementary labels1/34 and all six atlases were generated. Visual review
  prompted removal of overlapping endpoint annotations before production.
- Production validates2300 case records and80500 label slots. Independent
  reconstruction of8050 exported mean/sample-SD values has max difference0.
- Tracked-notebook output check and `git diff --check` passed. No GPU work,
  model imports, image/field arrays or metric evaluator calls are used.
- Production accepted in `releases/v001/`, with immutable `registry/v001.json`.
  Independent audit verified two35-page PDFs,70 PNG and70 SVG individual
  figures, six atlases in each PNG/PDF/SVG format and all180 output hashes.
  Output size approximately96.1MB; no source predictions were copied.
- Exact source-table parity: main19 case means/SDs unchanged. Source scalar
  hashes and field existence/size/mtime checks passed before acceptance.
  Label slots80500 include46 canonical exclusions; all23 Right-Vessel groups
  exclude exactly ranks35/58 and use98 cases on both axes.
- MR dense best-Dice lambdas across35 labels:0.3 for14,1 for10,3 for11.
  At matched lambda1/3, dense improves label-mean Dice in35/35 labels; HD95
  improves30/35 and31/35; ASD34/35 and35/35, respectively. Descriptive only.
- Visual checks covered Right-Vessel, hippocampus prefix comparison and the
  complete Dice atlas. Reports use official seg35 names and historical-DL
  markers. No registration/evaluation, source artifact changes or GPU actions.
