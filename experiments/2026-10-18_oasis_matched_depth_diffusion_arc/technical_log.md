# Technical log

## Limitations

Foreground displacement diffusion is a common diagnostic, not the exact
training objective. Training is unmasked and the SVF arm regularizes velocity.
The axis excludes lambda weighting. Earlier accepted fields/reports are read-only.

## Implementation

- Separate `scripts/reports/oasis_matched_depth_diffusion_arc.py` uses the
  established two-endpoint foreground finite-difference function and raw-mask
  orientation check, with explicit normalized-xyz conversion for spline fields.
- Source checks and canonical full-volume parity guard against layout/unit errors.
- Figures inherit existing family styles and original matched-depth y limits.
- Tests and native smoke/production outcomes will be recorded after execution.

## Verification/preflight

- Ten new CPU tests pass: translation/affine fields, both-endpoint mask,
  normalized XYZ scale, component permutation, common training-formula parity,
  invalid inputs, sample SD/completeness, heterogeneous CSV schemas and figures.
- Six unchanged five-family ARC regression tests pass.
- Initial test harness needed dataclass-compatible module registration; fixed
  only the new test import, not the shared numerical function.
- Preflight review caught heterogeneous old/new/DL CSV schemas. The identified
  CPU smoke was stopped before measurements completed; its external directory
  and interruption receipt are retained as `smoke_v1`. Union-column CSV writer
  and regression test added. Source reads were parallelized without changing
  computations. Corrected smoke uses separate `smoke_v2`.
- Corrected native rank0 smoke passed all52 depth/family/lambda cells, zero
  failures,41.72s including provenance checks. Largest full-volume parity
  absolute difference4.120206995e-8. Production CPU analysis launched with
  eight threads under UID:GID1009:1020; no GPU workers or registrations.
- With467 cached fields, resumed the same analysis at16 CPU threads on the
  256-logical-CPU host to reduce shared-storage latency. The verified owned
  CPU process alone was stopped; its original manifest/progress are retained
  in `RESUME_16_THREADS.json`. Cached fields are rehashed before reuse; no
  scientific settings or GPU jobs changed.

## Completed

- Accepted2026-10-18 20:46:37UTC (13:46PDT): all5200 unique saved fields,
  52 complete100-case groups and zero failures. Each depth plots31 settings;
  the1000 DL fields are shared, not rerun or counted as independent replications.
- All field hashes/finite geometry and unmasked diffusion parity passed;
  maximum absolute parity error1.6861267063e-7. The renderer verified520
  original five-metric mean/sample-SD values exactly to1e-12 tolerance.
- Independent NumPy recomputation verified728 mean/sample-SD values from
  serialized per-case/summary tables, including both diffusion metrics. All100
  ranks have52 rows and identical fixed-mask hashes/edge counts across methods.
- All output hashes passed. Both depths have identical axis limits/ticks and
  retain the accepted original y limits. Both PNGs visually inspected; six
  PNG/PDF/SVG exports are readable. All12 original ARC export hashes unchanged.
- Last resumed segment took1738.67s, including rechecking467 cached fields;
  this is not total uninterrupted analysis time and is not registration time.
- Accepted external companion:
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_bspline842_8421_foreground_diffusion_arc_v1/release_v1/`.
  Includes per-case CSV/JSON,52-row summary, metadata, hashes and acceptance.
