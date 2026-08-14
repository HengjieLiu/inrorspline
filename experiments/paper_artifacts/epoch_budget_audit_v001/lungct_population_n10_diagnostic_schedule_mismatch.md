# LungCT N=10 Diagnostic Schedule Mismatch

Date: 2026-08-12

Scope: record the schedule mismatch discovered in the 10 OASIS + 10 DIR-LAB
diagnostic rerun, especially the DIR-LAB/LungCT MR-D-BSCP schedule difference
between Panel A and Panel B.

## Short Answer

The 10-case diagnostic rerun was **not** uniformly standardized to
`400,400,800,900` for LungCT.

- OASIS MR-D-BSCP diagnostics are internally matched at 2500 total steps.
- DIR-LAB/LungCT Panel A registration/global-NCC MR-D-BSCP used
  `200,400,800,1000` = 2400 total steps.
- DIR-LAB/LungCT Panel B pTVReg field-fit MR-D-BSCP used
  `400,400,800,900` = 2500 total steps.

So the user's memory is partly correct: the N=10 diagnostic population was
rerun, and LungCT Panel B did use `400,400,800,900`; however LungCT Panel A
MR-D-BSCP still used the older `200,400,800,1000` schedule.

## Evidence

Primary folder:

```text
experiments/2026-08-02_fit_modes_population_n10/
```

Primary manifest:

```text
experiments/2026-08-02_fit_modes_population_n10/diagnostic_manifest.json
```

The manifest records the issue explicitly in `mr_d_bscp_schedule_warning`:

```json
{
  "status": "open_camera_ready_issue",
  "oasis": {
    "panel_a": "400,800,1300",
    "panel_b": "400,800,1300",
    "total_epochs": 2500,
    "matched": true
  },
  "dirlab": {
    "panel_a": "200,400,800,1000",
    "panel_a_total_epochs": 2400,
    "panel_b": "400,400,800,900",
    "panel_b_total_epochs": 2500,
    "matched": false
  },
  "preferred_camera_ready_fix": "new versioned DIR-LAB Panel A MR-D-BSCP N=10 cohort using 400,400,800,900; preserve the accepted 2400-epoch cohort"
}
```

The curve summary agrees:

```text
experiments/2026-08-02_fit_modes_population_n10/diagnostic_curve_summary.csv
```

Observed MR-D-BSCP diagnostic rows:

| Dataset | Panel | Rows | Epoch max | Points | Status |
|---|---:|---:|---:|---:|---|
| OASIS | A | 10 | 2500 | 251 | OK |
| OASIS | B | 10 | 2500 | 251 | OK |
| DIR-LAB/LungCT | A | 10 | 2400 | 241 | Mismatch |
| DIR-LAB/LungCT | B | 10 | 2500 | 935 or 2500 | OK |

The `points=935` in some DIR-LAB Panel B MR-D-BSCP rows is a diagnostic
sampling-density detail, not a shorter training budget. Those rows still report
`epoch_max=2500.0` and come from the `400,400,800,900` stage-history runs.

## What Each Panel Means

Panel A:

- Image registration / global-NCC diagnostic.
- Metric: `1 - NCC`.
- For DIR-LAB MR-D-BSCP, this uses the native/global-NCC registration branch:
  `direct_cp_32_16_8_4_global_ncc_fd_be`.
- Current DIR-LAB Panel A MR-D-BSCP N=10 diagnostic source uses
  `200,400,800,1000`.

Panel B:

- pTVReg field-fit diagnostic.
- Metric: `100 * sqrt(MSE / (target_power + 1e-12))`.
- For DIR-LAB MR-D-BSCP, this uses pTVReg field fitting with finest-domain
  diagnostic history.
- Current DIR-LAB Panel B MR-D-BSCP N=10 diagnostic source uses
  `400,400,800,900`.

## Why This Matters

The paper standard we now want for LungCT multistage MR-D-BSCP is:

```text
400,400,800,900 = 2500 total optimizer steps
```

The N=10 diagnostic rerun is not yet fully aligned with that standard because
DIR-LAB Panel A MR-D-BSCP uses:

```text
200,400,800,1000 = 2400 total optimizer steps
```

This creates two problems:

1. A strict "2500 epochs for all learned/optimized methods" statement is not
   true for the DIR-LAB Panel A MR-D-BSCP diagnostic cohort.
2. The two DIR-LAB MR-D-BSCP diagnostic panels are not schedule-matched:
   Panel A uses the old 2400-step schedule, while Panel B uses the intended
   2500-step schedule.

This is not an over-training advantage for Panel A. If anything, Panel A has
100 fewer steps. The issue is provenance and standardization, not an unfair
extra budget.

## Relationship To Figure 2

This diagnostic N=10 rerun is related to paper diagnostics, but it is not the
same thing as the current paper Figure 2 TRE/stdLogJ aggregate source.

The key answer to the user's question is:

- If asking about the **N=10 diagnostic rerun**, LungCT Panel A was **not**
  fixed to `400,400,800,900`; LungCT Panel B was.
- If asking about the **current paper Figure 2 TRE/stdLogJ artifact**, the
  LungCT MR-D-BSCP rows also still point to the older
  `200,400,800,1000` schedule in the currently audited source artifacts.

Therefore neither the current LungCT Panel A diagnostics nor the current
LungCT TRE/stdLogJ MR-D-BSCP result branch should be considered fully
standardized to `400,400,800,900` yet.

## Recommended Camera-Ready Fix

Create a new versioned DIR-LAB/LungCT Panel A MR-D-BSCP N=10 diagnostic cohort
using:

```text
method: direct_cp_32_16_8_4_global_ncc_fd_be
cases: DIR-LAB case01-case10
schedule: 32,16,8,4
downsample: 8,4,2,1
stage_epochs: 400,400,800,900
lr: 1e-3
seed: 1
diagnostic: global_ncc_diagnostic.csv
```

Preserve the existing accepted 2400-step cohort as historical provenance; do
not overwrite it.

After rerun:

1. Regenerate `diagnostic_curve_summary.csv`.
2. Regenerate all 10 DIR-LAB diagnostic figures.
3. Regenerate `diagnostic_manifest.json`.
4. Verify that `mr_d_bscp_schedule_warning.dirlab.matched` becomes `true`.
5. Verify that DIR-LAB MR-D-BSCP Panel A reports `epoch_max=2500.0`.
6. Keep a manifest note that the new cohort supersedes, but does not delete,
   the original `200,400,800,1000` diagnostic cohort.

## Minimal Acceptance Criteria

- DIR-LAB Panel A MR-D-BSCP has 10 rows, one per case.
- All 10 DIR-LAB Panel A MR-D-BSCP rows use `400,400,800,900`.
- All 10 DIR-LAB Panel A MR-D-BSCP rows report total budget 2500.
- OASIS schedule remains unchanged at `400,800,1300`.
- DIR-LAB Panel B remains unchanged at `400,400,800,900`.
- Existing 2400-step DIR-LAB Panel A artifacts remain traceable.

## Bottom Line

The N=10 diagnostic rerun did happen, but LungCT was only half standardized:
Panel B already uses `400,400,800,900`, while Panel A still uses
`200,400,800,1000`. For camera-ready consistency, the needed rerun is the
DIR-LAB/LungCT Panel A MR-D-BSCP N=10 diagnostic cohort with
`400,400,800,900`.
