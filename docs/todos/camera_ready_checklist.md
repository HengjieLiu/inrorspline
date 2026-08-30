---
doc_type: camera_ready_checklist
status: active
---

# Camera-Ready Checklist

Use this checklist before freezing camera-ready experiments, figures, captions,
reviewer responses, and manuscript claims. An unchecked blocking item must be
resolved or explicitly retained as a reported limitation.

## Blocking experiment consistency

- [x] **Resolve the OASIS Panel A metric-domain mismatch in the population
  fit-modes diagnostic figure.**
  - v002 uses fixed T00 lung-mask `1-NCC` for DIR-LAB Panel A but still uses
    full-image global `1-NCC` for OASIS Panel A.
  - Preferred camera-ready fix: rerun OASIS Panel A diagnostics for the seed-3
    first10 cohort with paired full-image and fixed raw `image > 0` masked NCC
    curves, then plot the fixed-mask metric in v003.
  - Acceptance: all 40 OASIS Panel A records complete; each curve has 251
    finite fixed-mask diagnostics through epoch 2500; v003 replaces exactly
    OASIS Panel A and preserves DIR-LAB v002 Panel A plus all Panel B rows.
  - Completed 2026-08-14: `40/40` OASIS Panel A records passed and the v003
    figure family was regenerated from a 160-row source index. See
    `experiments/paper_artifacts/fit_modes_population_diagnostics/v003_masked_panelA_both_datasets/validation_report.md`.

- [x] **Refresh DIR-LAB Panel A single-stage BSCP alphas for regularity-matched
  Figure 3 diagnostics.**
  - v003 used fixed-domain metrics but retained the same nominal FD-BE alpha
    for INR-BSCP and D-BSCP. The DIR-LAB Pareto sweep showed this did not give
    comparable foreground `SD(log J)`.
  - Camera-ready v004 uses exact-compatible reuse of the accepted all-10
    FD-BE cps4 alpha-landscape curves: INR-BSCP cps4 `a=20000` and D-BSCP cps4
    `a=50000`.
  - Acceptance: v004 replaces exactly 20 DIR-LAB Panel A rows, preserves the
    other 140 v003 rows, validates 251 finite fixed-T00-lung-mask diagnostics
    for every replacement, and regenerates the Figure 3 artifact family with
    Panel A axis `1e-2–1e0`.
  - Completed 2026-08-14: see
    `experiments/paper_artifacts/fit_modes_population_diagnostics/v004_dirlab_regularity_matched_panelA/validation_report.md`.

- [x] **Resolve the DIR-LAB MR-D-BSCP schedule asymmetry in the population
  fit-modes experiment.**
  - Historical Panel A registration schedule: cps `32-16-8-4`, downsample
    `8-4-2-1`, epochs `200/400/800/1000` = **2400 total**.
  - Accepted replacement Panel A and current Panel B field-fitting schedule:
    cps `32-16-8-4`, downsample
    `8-4-2-1`, epochs `400/400/800/900` = **2500 total**.
  - OASIS does not have this asymmetry: both panels use `400/800/1300` =
    2500 total.
  - Preferred camera-ready fix: run a new, versioned DIR-LAB Panel A
    MR-D-BSCP N=10 cohort with `400/400/800/900`. This matches Panel B exactly
    and gives every method in Panel A a 2500-epoch budget.
  - Budget-only alternative: `200/400/800/1100` preserves the existing Panel A
    transition locations and reaches 2500 epochs, but it does **not** fix the
    cross-panel schedule mismatch and must not be described as schedule-matched.
  - Preserve the accepted `200/400/800/1000` results as historical/sensitivity
    results; do not overwrite or relabel them.
  - Acceptance: all 10 replacement runs complete; stage boundaries, final
    epoch, source curves, and endpoints validate; the result is compared with
    the historical 2400-epoch cohort before choosing the primary paper result.
  - Completed 2026-08-13: `10/10` replacements passed with 251 full-crop
    diagnostics each and zero unresolved failures. See
    `experiments/2026-08-13_fit_modes_dirlab_panelA_schedule_refresh/acceptance_report.md`.

## Required downstream updates after the schedule fix

- [x] Reaggregate DIR-LAB Panel A MR-D-BSCP endpoints and population statistics
  into the focused replacement and comparison indexes. Main population-index
  integration remains part of figure step 4.
- [x] Regenerate the schedule-matched OASIS10/DIR-LAB10 supplementary diagnostic
  figures from source; update the versioned source index, manifests, checksums,
  and visual-review record.
- [x] Design and regenerate the separate compact main-paper figure from the
  accepted schedule-matched source index.
  - Completed 2026-08-15 UTC: final camera-ready Figure 2 is
    `experiments/paper_artifacts/pareto_figure/v013_final_dirlab4dct_axes_labels/`.
  - Camera-ready outputs:
    `pareto_four_panel_final_dirlab4dct_no_annotations.png` and
    `pareto_four_panel_final_dirlab4dct_no_annotations.pdf`.
  - Review-only output:
    `pareto_four_panel_final_dirlab4dct_annotated_review.png`.
  - Source validation: OASIS rows match `v003_candidate_oasis100`; DIR-LAB
    rows match `v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape`; no
    masked-loss rows or paths are included. All four panels use the unified
    x-axis label `Foreground SDlogJ`.
- [ ] Update Methods, captions, legends, and reviewer response with the exact
  cps, downsample, stage-epoch schedule, and total optimization budget.
- [x] Resolve the claim blocker: the accepted replacement cohort now supports
  identical DIR-LAB MR-D-BSCP schedules and equal 2500-epoch budgets in the
  versioned camera-ready index; retain the historical-schedule caveat for the
  original index.
- [ ] State whether the historical 2400-epoch result is shown as a sensitivity
  analysis or retained only as provenance.

## Final fit-modes paper audit

- [ ] Verify that OASIS is described as a prespecified, reproducibly shuffled
  N=10 subset—not an unconstrained random sample or the previous N=100 cohort.
- [ ] Retain the OASIS N=10 versus requested N=100 limitation.
- [ ] State explicitly that Panel B is target-field representation/fitting, not
  registration accuracy or generalization.
- [ ] Show the actual final-spacing settings: OASIS `cps2` versus `8-4-2`, and
  DIR-LAB `cps4` versus `32-16-8-4`.
- [ ] Confirm every camera-ready figure is regenerated from its committed
  source and not from an untracked raster patch.
