# Technical log

## Limitations

Exploratory test-case tuning; ellipses are marginal sample SD, not statistical
confidence intervals. SD(log J) excludes nonpositive determinants; folding is
reported separately. Baselines are saved learned fields, not spline approximations.

## Implementation

Reuses audited saved-field loading, canonical accuracy evaluation and common
Jacobian metrics. Source40 raw-NCC rows remain immutable. Figures follow the
existing axis-aligned2SD ellipse convention with a separate folding companion.
Preflight, numerical verification and visual inspection are recorded below after
execution; this specification does not itself imply successful completion.

## Preflight

- Four CPU contracts passed: sample SD and subset means, missing/duplicate/
  nonfinite rejection, exact axis-aligned2SD ellipse geometry and PNG/SVG/PDF
  export. Whitespace audit passed.
- Rank0 GPU smoke reproduced the saved canonical VFA and SITReg accuracy
  metrics exactly, including SITReg Dice/HD95/ASD
  `0.7987193558/2.0613193370/0.4230641215`.
- Main first10 posthoc evaluation launched on GPU0 as UID:GID1009:1020.
  Existing registration fields and GPU jobs are not changed.

## Completion and visual QA

- All20 original baseline fields evaluated; all VFA accuracy values match P40.
  The 40 raw-NCC rows preserve their original accuracy and regularity values.
  Independent standard-library calculations reproduce all12 mean/sample-SD rows
  within1e-12. Source-report hash remains unchanged.
- SITReg first10 mean/sample SD Dice `0.821545854/0.024232248`, HD95
  `1.715898344/0.273759216` mm, symmetric ASD `0.350896166/0.075233110` mm.
  Foreground SD(log J) mean0.569473068; nonpositive-J percentage0.000423009%.
- Generated and visually inspected three-panel ARC and folding companion
  figures, each in PNG/SVG/PDF. Added a small left-axis margin to the folding
  plot so SITReg's near-zero-folding marker is not clipped. This changes only
  plot padding, not data, statistics or ellipse sizes.
