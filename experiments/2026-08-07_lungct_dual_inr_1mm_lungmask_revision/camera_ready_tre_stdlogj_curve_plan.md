# LungCT DUAL-INR Camera-Ready TRE/stdlogJ Curve Plan

## Goal

Add a small number of defensible DUAL-INR-family curves to the camera-ready
DIR-Lab TRE versus stdlogJ figure.

This campaign is figure-driven, not an exhaustive hyperparameter search. Each
candidate curve point must use all 10 DIR-Lab 4DCT cases so that the plotted
mean TRE and mean stdlogJ are cohort-level quantities, not selected-case
screening results.

## Main Question

Can controlled DUAL-INR variants provide one or more useful additional lines on
the TRE/stdlogJ curve when the image domain, mask domain, schedule, and loss are
explicitly tagged?

The intended output is not "find the globally best DUAL-INR setting." The
intended output is:

- choose a few scientifically interpretable DUAL-INR-family curves;
- report their all-case TRE and regularity tradeoff;
- decide which curve points are worth adding to the final DIR-Lab figure;
- avoid wasting runs on implausibly loose or implausibly rigid settings.

## Fixed Cohort And Protocol

- Dataset: DIR-Lab 4DCT cases `01` through `10`.
- Direction: T50 moving image to T00 fixed image.
- Image domain: canonical pTVReg 1 mm isotropic images.
- Image masking: use the unmasked 1 mm image intensities.
- Sampling mask: fixed T00 1 mm lung mask.
- Metric mask: fixed T00 1 mm lung mask.
- Do not use moving lung masks for optimization.
- Do not use semantic, lobe, or segmentation loss.
- Large run artifacts stay outside Git under `<container-data-root>/inrorspline/runs/...`.
- Small review summaries, CSV tables, and Markdown reports may be written under
  this experiment folder.

Interpretation must preserve protocol tags. These rows should not be presented
as clean architecture-only comparisons against original IDIR, pTVReg, SINR, or
DirectCP rows unless preprocessing, mask, metric, interpolation, and loss
conventions are explicitly matched or noted.

## Candidate Curves

### PL1D: Image-Pyramid DUAL, Native DUAL Regularization

- Schedule: explicit image pyramid `[8,4,2,1]`.
- Stage lengths: `[400,400,800,900]`.
- Network: released two-branch DUAL SIREN.
- Branch activation:
  - stage 0, factor 8: coarse SIREN only;
  - stages 1-3, factors 4, 2, 1: coarse + fine SIREN.
- Loss:
  - image similarity on fixed-lung-mask samples;
  - native DUAL Laplacian plus negative-Jacobian style regularization;
  - no semantic/lobe loss.

The default point already exists from the PL1D full-cohort run and should be
reused when the implementation and protocol identity match.

### PL1B: Image-Pyramid DUAL, Bending-Energy Regularization

- Schedule: explicit image pyramid `[8,4,2,1]`.
- Stage lengths: `[400,400,800,900]`.
- Network: released two-branch DUAL SIREN.
- Loss:
  - image similarity on fixed-lung-mask samples;
  - IDIR-type bending energy on the total DUAL displacement;
  - no semantic/lobe loss.

The default point already exists from the PL1B full-cohort run and should be
reused when the implementation and protocol identity match.

### PL1A: Original-Blur DUAL, Native DUAL Regularization

- Schedule: original released DUAL blur schedule `[4,2,0]`.
- Transition rule: original/adaptive convergence transition.
- Total budget: `3000` steps.
- Network: released two-branch DUAL SIREN.
- Loss:
  - image similarity on fixed-lung-mask samples;
  - native DUAL Laplacian plus negative-Jacobian style regularization;
  - no semantic/lobe loss.

The default point already exists from the PL1A full-cohort run and should be
reused when the implementation and protocol identity match.

### PL1C: Original-Blur DUAL, pTVReg-Style Regularization

- Schedule: original released DUAL blur schedule `[4,2,0]`.
- Transition rule: original/adaptive convergence transition.
- Total budget: `3000` steps.
- Network: released two-branch DUAL SIREN.
- Loss:
  - sampled pTVReg-style local cross-correlation image loss;
  - pTVReg-style physical-unit TV/Jacobian regularization on the total DUAL
    displacement;
  - no semantic/lobe loss.

This is the main new loss-control variant. It is intended to make a DUAL-INR
row closer to the best MR-D-BSCP pTVReg-loss comparison line while preserving
the DUAL SIREN architecture and original blur schedule.

## Compact Weight Sweep

The sweep should cover plausible camera-ready figure points, not extreme
regularization limits. Do not use a broad `1e-4` to `1e5` decade sweep unless a
later report shows the compact range misses the useful trend.

### Native DUAL-Loss Variants: PL1D and PL1A

Use five regularization multipliers:

```text
multiplier = [0.01, 0.1, 1, 10, 100]
```

For each multiplier `m`:

```text
coarse jac/lap weight = 0.01  * m
fine   jac/lap weight = 0.001 * m
```

This preserves the original 10:1 coarse/fine native-regularization ratio.

### Bending-Energy Variant: PL1B

Use the same five multipliers:

```text
multiplier = [0.01, 0.1, 1, 10, 100]
```

For each multiplier `m`:

```text
bending weight = 10.0 * m
```

### pTVReg-Style Variant: PL1C

Center the sweep on the known strong pTVReg-loss reference weight `0.055`:

```text
regularization weight = [0.0055, 0.017, 0.055, 0.17, 0.55]
```

This is approximately:

```text
[reference / 10, reference / 3, reference, reference * 3, reference * 10]
```

The exact stored metadata should record both the absolute weight and its
multiplier relative to `0.055`.

## Run Matrix

The compact all-case sweep contains:

```text
4 variants x 5 weights x 10 cases = 200 curve-point jobs
```

Existing default-weight full-cohort results may be reused:

- PL1D, multiplier `1`;
- PL1B, multiplier `1`;
- PL1A, multiplier `1`.

If those rows are reused, the number of new jobs becomes:

```text
200 total planned jobs - 30 reusable jobs = 170 new jobs
```

PL1C has no existing default full-cohort result and should be run for all five
weights unless an implementation smoke test fails.

Use all four available GPUs for execution, but keep the run manifest explicit:
each job must record case, variant, schedule, loss branch, absolute weights,
multiplier, image-domain tag, sampling-mask tag, metric-mask tag, code version,
Docker image, and output paths.

## Staged Execution

### Stage 0: Infrastructure And Dry Run

Purpose: make the campaign reproducible before launching expensive jobs.

Required work:

- add or confirm PL1C implementation;
- add explicit regularization-weight override support for PL1D, PL1B, PL1A,
  and PL1C;
- add campaign launch support for the compact sweep matrix;
- add aggregation grouped by variant, schedule, loss branch, and weight;
- add acceptance checks for all intended protocol tags;
- update the experiment registry before real runs;
- write the resolved dry-run manifest.

Report:

```text
stage0_implementation_and_manifest_report.md
```

The report should include:

- exact variants and weights planned;
- number of jobs to run versus reuse;
- paths to reused PL1D, PL1B, and PL1A default results;
- dry-run manifest summary;
- tests and smoke commands run;
- any known limitations before launch.

### Stage 1: PL1C Smoke And Default Point

Purpose: verify the new pTVReg-style loss branch before a full sweep.

Minimum smoke:

- PL1C, case `01`, weight `0.055`;
- short or standard run depending on implementation risk;
- confirm finite image loss, finite regularization loss, valid deformation
  field, and completed metric calculation.

If the smoke passes, run PL1C weight `0.055` on all 10 cases.

Report:

```text
stage1_pl1c_default_report.md
```

Decision:

- continue to Stage 2 if PL1C completes all 10 cases and produces finite TRE
  and regularity metrics;
- fix implementation or scaling if PL1C is unstable, folded everywhere, or
  obviously mis-scaled.

### Stage 2: Compact All-Case Sweep

Purpose: generate the candidate TRE/stdlogJ curves.

Run all 10 cases for every unreused weight point:

- PL1D: multipliers `[0.01,0.1,1,10,100]`;
- PL1B: multipliers `[0.01,0.1,1,10,100]`;
- PL1A: multipliers `[0.01,0.1,1,10,100]`;
- PL1C: weights `[0.0055,0.017,0.055,0.17,0.55]`.

Default points may be imported from previous reports only if:

- protocol is `PL`;
- image domain is pTVReg 1 mm;
- sampling and metric masks are fixed T00 1 mm lung masks;
- semantic/lobe loss is disabled;
- schedule identity matches the candidate curve;
- loss branch and default weights match exactly;
- posthoc regularity is complete.

Report:

```text
stage2_compact_all_case_sweep_report.md
```

The report should include:

- all-case mean and standard deviation of native snap TRE;
- all-case mean and standard deviation of subvoxel TRE, if available;
- all-case mean and standard deviation of foreground stdlogJ;
- all-case mean and total rate of nonpositive Jacobian voxels;
- per-case TRE table for every candidate point;
- curve table sorted by mean TRE and by stdlogJ;
- comparison against pTVReg and best MR-D-BSCP pTVReg-loss reference rows;
- a figure-ready CSV with one row per curve point.

### Stage 3: Local Refinement Only If Needed

Purpose: improve curve resolution near a useful operating point.

Do not launch refinement automatically. Use Stage 2 to decide whether it is
needed.

Eligible refinement patterns:

```text
best / 3
best
best * 3
```

Only refine a variant if at least one compact-sweep point is promising for the
camera-ready plot. A point is promising if it satisfies most of:

- mean TRE is competitive with pTVReg or MR-D-BSCP pTVReg-loss references;
- stdlogJ is in a useful range for the existing curve;
- nonpositive Jacobian rate is zero or acceptably low;
- the curve adds interpretation not already covered by existing methods;
- case-wise behavior is not driven by one outlier case.

Possible examples:

- If PL1C at `0.055` is best, refine around `[0.017,0.055,0.17]`; these may
  already exist from Stage 2, so no extra runs are needed.
- If PL1B at multiplier `10` is best, add PL1B multipliers
  `[10/3,10,30]`, skipping duplicate `10`.
- If PL1D or PL1A has no competitive or interpretable point, do not refine it.

Report:

```text
stage3_local_refinement_report.md
```

The report should explain:

- which variants were refined;
- why other variants were not refined;
- the added weights;
- whether refinement changed the recommended camera-ready curve points.

### Stage 4: Final Figure Recommendation

Purpose: select the rows that should be plotted in the camera-ready TRE/stdlogJ
figure.

Report:

```text
final_camera_ready_tre_stdlogj_curve_recommendation.md
```

The report should include:

- final recommended DUAL-INR-family curves;
- exact curve labels for the plot legend;
- exact row identifiers and run roots;
- final figure-ready CSV path;
- final all-case aggregate table;
- per-case TRE table for selected rows;
- regularity summary for selected rows;
- comparison against pTVReg and best MR-D-BSCP pTVReg-loss references;
- short interpretation text suitable for the paper or response notes.

## Selection Rules For Camera-Ready Curves

Prefer including a curve if it provides a distinct scientific comparison:

- schedule effect: PL1D image pyramid versus PL1A original blur;
- regularizer effect: PL1A native DUAL loss versus PL1C pTVReg-style loss;
- architecture/loss interaction: PL1C versus MR-D-BSCP pTVReg-loss reference;
- smoothness tradeoff: a visible TRE/stdlogJ trend across plausible weights.

Avoid including a curve if:

- all points are dominated by existing methods in both TRE and stdlogJ;
- the curve is unstable or has unacceptable folding;
- the curve duplicates another DUAL-INR curve without adding interpretation;
- performance is explained by a single case rather than all-case behavior;
- the protocol differs in a way that would make the camera-ready plot confusing.

## Baseline Comparisons

Each report should include the same reference rows when available:

- pTVReg reference row under the matched DIR-Lab/pTVReg convention;
- best MR-D-BSCP pTVReg-loss row, currently weight `0.055`;
- existing PL1D, PL1B, and PL1A default rows when they are reused.

The report must state whether the comparison is matched in:

- image crop/resolution;
- optimization mask;
- metric mask;
- TRE convention;
- interpolation convention;
- regularization family.

## Acceptance Criteria

A curve point is technically accepted if:

- all 10 cases either complete or are explicitly marked failed;
- completed cases have finite TRE and finite regularity metrics;
- expected artifacts are present;
- protocol and schedule tags match the intended variant;
- loss branch and weights match the intended point;
- sampling mask and metric mask are fixed T00 1 mm lung masks;
- semantic/lobe loss is disabled;
- posthoc regularity is complete;
- no raw data, checkpoints, dense fields, or medical volumes are written to Git.

TRE magnitude is not a technical completion gate. TRE and stdlogJ are used only
for scientific selection after acceptance.

## Recommended Immediate Next Step

Implement Stage 0 only first:

1. Add PL1C and weight-override support.
2. Generate the dry-run manifest for all compact-sweep jobs.
3. Write `stage0_implementation_and_manifest_report.md`.
4. Run one PL1C smoke job at weight `0.055`.

Only after the smoke and manifest are reviewed should the all-GPU Stage 2 sweep
be launched.
