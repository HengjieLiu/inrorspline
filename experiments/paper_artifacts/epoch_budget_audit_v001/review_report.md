# Paper Figure Epoch-Budget Audit v001

Date: 2026-08-12

Scope: audit paper-facing Figure 1 and Figure 2 artifacts for the intended
standard of 2500 optimizer steps/epochs per learned/optimized method. For
multistage methods, the budget is the sum of stage epochs.

## Executive Summary

There is a real 2400-step inconsistency, but it is localized.

- OASIS paper-facing rows inspected here are consistent with a 2500-step
  budget.
- LungCT single-stage rows inspected here are consistent with a 2500-step
  budget.
- LungCT Figure 1 Panel B field-fit MR-D-BSCP uses `400+400+800+900 = 2500`
  and is consistent.
- LungCT Figure 1 Panel A image-registration/global-NCC MR-D-BSCP uses
  `200+400+800+1000 = 2400` and is inconsistent.
- LungCT Figure 2 TRE/stdLogJ MR-D-BSCP rows, both IDIR-loss and pTVReg-loss
  variants, use the same `200+400+800+1000 = 2400` schedule and are
  inconsistent with a strict 2500-step paper standard.

This is not an unfair extra-budget advantage for MR-D-BSCP; it is the opposite:
the affected LungCT MR-D-BSCP rows have 100 fewer optimizer steps than the
2500-step rows. But if the paper states all learned methods use 2500 epochs,
the statement is currently false for those LungCT 4-stage result rows.

## Budget Rule Used

- Single-stage run: `epochs`.
- Multistage run: `sum(stage_epochs)`.
- External references such as pTVReg, SITReg, VoxelMorph, TransMorph, VFA, and
  Greedy are not governed by this epoch-budget standard.

## Figure 1 Audit

Current paper-facing fitting figure source:
`experiments/paper_artifacts/fitting_figure/v002_candidate/fit_modes_standardized_panelAB_curve_summary.csv`

### Panel A: Image-Registration / Global-NCC Curves

| Dataset/case | Method | Source evidence | Budget | Status |
|---|---:|---|---:|---|
| OASIS rank-9 | INR-Dense | `epoch_max=2500.0` | 2500 | OK |
| OASIS rank-9 | INR-BSCP | run path contains `e2500` | 2500 | OK |
| OASIS rank-9 | D-BSCP | run path contains `e2500` | 2500 | OK |
| OASIS rank-9 | MR-D-BSCP | run path contains `e400-800-1300` | 2500 | OK |
| LungCT case01 | INR-Dense | `epoch_max=2500.0` | 2500 | OK |
| LungCT case01 | INR-BSCP | `epoch_max=2500.0` | 2500 | OK |
| LungCT case01 | D-BSCP | `epoch_max=2500.0` | 2500 | OK |
| LungCT case01 | MR-D-BSCP | run path contains `e200-400-800-1000` | 2400 | Needs fix |
| LungCT case08 | INR-Dense | `epoch_max=2500.0` | 2500 | OK |
| LungCT case08 | INR-BSCP | `epoch_max=2500.0` | 2500 | OK |
| LungCT case08 | D-BSCP | `epoch_max=2500.0` | 2500 | OK |
| LungCT case08 | MR-D-BSCP | run path contains `e200-400-800-1000` | 2400 | Needs fix |

The problematic rows are lines 9 and 13 of the current curve-summary CSV.

### Panel B: Field-Fit Curves

| Dataset/case | Method | Source evidence | Budget | Status |
|---|---:|---|---:|---|
| OASIS rank-9 | INR-Dense | run path contains `e2500` | 2500 | OK |
| OASIS rank-9 | INR-BSCP | run path contains `e2500` | 2500 | OK |
| OASIS rank-9 | D-BSCP | run path contains `e2500` | 2500 | OK |
| OASIS rank-9 | MR-D-BSCP | run path contains `e400-800-1300` | 2500 | OK |
| LungCT case01 | INR-Dense | run path contains `e2500` | 2500 | OK |
| LungCT case01 | INR-BSCP | run path contains `e2500` | 2500 | OK |
| LungCT case01 | D-BSCP | run path contains `e2500` | 2500 | OK |
| LungCT case01 | MR-D-BSCP | run path contains `e400-400-800-900` | 2500 | OK |
| LungCT case08 | INR-Dense | run path contains `e2500` | 2500 | OK |
| LungCT case08 | INR-BSCP | run path contains `e2500` | 2500 | OK |
| LungCT case08 | D-BSCP | run path contains `e2500` | 2500 | OK |
| LungCT case08 | MR-D-BSCP | run path contains `e400-400-800-900` | 2500 | OK |

Panel B is therefore not the source of the 2400-step problem.

## Figure 2 Audit

### OASIS Figure 2 / Table Rows

Current source rows:
`experiments/paper_artifacts/pareto_figure/v002_candidate/oasis20_pareto_source_rows.csv`

The OASIS aggregate source CSV does not retain per-run `epochs` or
`stage_epochs` columns, so it is less directly auditable than the LungCT source
rows. However, the paper-facing fitting/source artifacts and artifact standard
show:

- single-stage OASIS source run names use `e2500`;
- OASIS MR-D-BSCP uses `e400-800-1300`, total 2500;
- no 2400-step OASIS paper-facing signature was found in the audited Figure 1
  and Figure 2 source surfaces.

Status: no OASIS 2400-step paper-facing issue found.

### LungCT Figure 2 / TRE-stdLogJ Rows

Current source rows:
`experiments/paper_artifacts/pareto_figure/v002_candidate/lungct10_pareto_source_rows.csv`

| Method in plot/table | Method id | Source evidence | Budget | Status |
|---|---|---|---:|---|
| INR-Dense (IDIR loss) | `idir_be_default_lungmask` | summary JSON has `epochs=2500`; aggregate paths contain `e2500` | 2500 | OK |
| INR-BSCP (pTVReg loss) | `sinr_cps4_ptv_lcc_isotv` | `summary_runs.csv` has `epochs=2500` | 2500 | OK |
| D-BSCP (pTVReg loss) | `direct_cp_cps4_ptv_lcc_isotv` | `summary_runs.csv` has `epochs=2500` | 2500 | OK |
| MR-D-BSCP (IDIR loss) | `direct_cp_32_16_8_4_global_ncc_fd_be` | case summary paths contain `e200-400-800-1000`; JSON stores `stage_epochs=[200,400,800,1000]` | 2400 | Needs fix |
| MR-D-BSCP (pTVReg loss) | `direct_cp_32_16_8_4_ptv_lcc_isotv` | case summary paths contain `e200-400-800-1000`; JSON stores `stage_epochs=[200,400,800,1000]` | 2400 | Needs fix |
| pTVReg reference | `ptvreg_reference` | external uploaded pTVReg deformation | N/A | Exempt |

Affected plotted Figure 2 curve rows:

- `direct_cp_32_16_8_4_global_ncc_fd_be`: all eight plotted alpha values
  `[0, 10, 100, 1000, 3000, 10000, 30000, 100000]`.
- `direct_cp_32_16_8_4_ptv_lcc_isotv`: all seven plotted weights
  `[0.0011, 0.011, 0.033, 0.11, 0.33, 1.1, 11]`.

Affected best/table points:

- `MR-D-BSCP` / `MR-D-BSCP (IDIR loss)`, selected `alpha=10000`.
- `MR-D-BSCP (pTVReg loss)`, selected `w=0.11`.

## Code/Launcher Sources of the Inconsistency

The LungCT no-sweep/multistage runner defaults to:

```text
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
```

That total is 2400. The same default is also mirrored in the LungCT 4-stage
sweep launchers.

Representative files:

- `scripts/experiments/lungct_no_sweep_repro/run_one.py`
- `scripts/experiments/lungct_no_sweep_repro/launch_directcp32_global_ncc_fdbe_sweep.py`
- `scripts/experiments/lungct_no_sweep_repro/launch_reg_sweep_idir_directcp32.py`

## Impact

Scientific/fairness impact:

- The affected LungCT MR-D-BSCP rows receive 4% fewer optimizer steps than the
  intended 2500-step standard.
- This does not create an over-training advantage for MR-D-BSCP.
- It does create a methods/provenance inconsistency and can make the paper
  statement "2500 epochs for all learned methods" inaccurate.

Artifact impact:

- Figure 1 Panel A has two affected plotted curves: LungCT case01 and case08
  MR-D-BSCP global-NCC curves.
- Figure 1 Panel B appears clean.
- Figure 2 LungCT TRE/stdLogJ curve/table rows are affected for both
  multistage LungCT MR-D-BSCP loss variants.
- OASIS Figure 2 appears clean from the paper-facing artifacts inspected here.

## Recommended Fix

Use a strict 2500-step standard for learned/optimized paper rows.

Minimal schedule-only correction for LungCT 4-stage MR-D-BSCP:

```text
old: 200,400,800,1000 = 2400
new: 200,400,800,1100 = 2500
```

This keeps the same image pyramid and adds the missing 100 steps to the final
full-resolution stage. It is the least disruptive correction because it does
not change the number of stages, control-point levels, downsample factors, loss,
or learning rate.

Recommended reruns:

1. Figure 1 Panel A sample curves:
   - LungCT case01, `direct_cp_32_16_8_4_global_ncc_fd_be`, `alpha=10000`.
   - LungCT case08, `direct_cp_32_16_8_4_global_ncc_fd_be`, `alpha=10000`.

2. Figure 2 LungCT MR-D-BSCP IDIR-loss curve:
   - all 10 cases;
   - alphas `[0, 10, 100, 1000, 3000, 10000, 30000, 100000]`;
   - schedule `200,400,800,1100`.

3. Figure 2 LungCT MR-D-BSCP pTVReg-loss curve:
   - all 10 cases;
   - weights `[0.0011, 0.011, 0.033, 0.11, 0.33, 1.1, 11]`;
   - schedule `200,400,800,1100`.

If runtime is constrained, the minimum publication-safe rerun would be the
selected best/table points only:

- IDIR-loss `alpha=10000`, all 10 cases;
- pTVReg-loss `w=0.11`, all 10 cases.

But for the TRE/stdLogJ curves themselves, rerunning only the best points would
leave mixed-budget curve endpoints. For a clean camera-ready Figure 2, rerun the
full plotted curves.

## Recommended Artifact/Validation Changes

Add explicit budget provenance to paper artifact builders:

- `epoch_budget_total`
- `epochs`
- `stage_epochs`
- `budget_status` with values like `matches_2500`, `nonstandard_2400`,
  `external_reference`, `unknown_source`

Validation should warn or fail for learned/optimized paper rows with total
budget other than 2500, excluding external references.

For aggregate CSVs, do not rely only on run-name parsing. Copy the budget fields
from per-case `summary.json` into case rows before cohort aggregation, then keep
the cohort-level unique budget signature in source rows/manifests.

## Bottom Line

Yes, your concern is correct. The main issue is the LungCT 4-stage MR-D-BSCP
result branch, not the OASIS branch and not most single-stage rows. The cleanest
paper-standard fix is to rerun the affected LungCT 4-stage MR-D-BSCP rows with
`stage_epochs=200,400,800,1100` and regenerate Figure 1 Panel A plus Figure 2
LungCT source artifacts from those runs.
