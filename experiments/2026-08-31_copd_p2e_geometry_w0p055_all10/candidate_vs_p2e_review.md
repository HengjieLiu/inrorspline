---
report_type: isolated_candidate_comparison
status: review_required_before_external_reporting
metric: native-domain snap-to-voxel landmark TRE (mm)
---

# COPD P2e-geometry `w=0.055` candidate versus frozen P2e

This is an isolated review of the completed all-ten-case COPD candidate.  It
does **not** modify or replace the frozen P2e `w=0.11` row in the cross-dataset
summary.  Both columns use the same direction (`eBHCT -> iBHCT`), canonical
pTV-style 1-mm landmark crop, direct cubic B-spline control optimization, and
native-crop snap-to-voxel landmark TRE.

The sole intended scientific change is isoTV weight: `0.11 -> 0.055`.  Controls
remain `64 -> 32 -> 16 -> 8 -> 4`; image scales are `16 -> 8 -> 4 -> 2 -> 1`;
stage epochs are `400,400,400,400,900`; Adam LR is `1e-3`; seed is `1`.

Each case cell is mean +/- sample SD across 300 landmarks.  The final row is
the unweighted mean +/- **sample SD** across the ten case means.  `Delta` is
candidate minus frozen P2e; negative favors the candidate.

| COPD case | Frozen P2e `w=0.11` (mm) | Candidate `w=0.055` (mm) | Delta (mm) |
| ---: | ---: | ---: | ---: |
| 01 | 0.7141 +/- 0.8340 | 0.7343 +/- 0.8667 | +0.0203 |
| 02 | 3.3480 +/- 5.5756 | 3.1162 +/- 5.3751 | -0.2318 |
| 03 | 0.8232 +/- 0.7868 | 0.8407 +/- 0.8064 | +0.0175 |
| 04 | 0.7325 +/- 0.9074 | 0.7986 +/- 1.0125 | +0.0661 |
| 05 | 0.7404 +/- 1.0838 | 0.7671 +/- 1.0608 | +0.0267 |
| 06 | 0.8511 +/- 1.4799 | 0.9279 +/- 1.7268 | +0.0768 |
| 07 | 0.8362 +/- 1.0649 | 0.7997 +/- 0.9668 | -0.0365 |
| 08 | 0.8196 +/- 1.2626 | 0.9721 +/- 1.7122 | +0.1526 |
| 09 | 0.8512 +/- 1.9594 | 0.6253 +/- 1.0161 | -0.2259 |
| 10 | 0.7886 +/- 0.7683 | 0.9190 +/- 1.4659 | +0.1304 |
| **Ten-case mean +/- sample SD** | **1.0505 +/- 0.8088** | **1.0501 +/- 0.7331** | **-0.0004 +/- 0.1325** |

The candidate is lower in three of ten cases (`02`, `07`, `09`) and higher in
seven.  Its nearly identical mean is driven by trade-offs across cases, not a
uniform improvement.  The reduced across-case SD is mainly associated with
smaller errors in cases `02` and `09`; it is not evidence that the candidate
has been selected.

## Completion and validation

- One-epoch-per-stage case-01 smoke completed first in a separate root.
- Full cases `01-10` then completed sequentially on GPU `0` only.
- Each summary records the literal `ptv_isotv_weight=0.055`, all frozen stage
  settings, 300 landmarks, finite native/registration-grid TREs, and a finite
  final dense field.
- The generated [`review_report.md`](review_report.md) and its CSV/JSON files
  are candidate-only artifacts.  Its aggregate helper uses population SD; the
  table above deliberately recomputes the stated sample-SD convention from
  case means.

## Sources

- Candidate case values: [`copd_case_rows.csv`](copd_case_rows.csv).
- Frozen P2e values: [existing COPD comparison CSV](../2026-08-30_copd_ptvreg_matlab_vs_mrdbscp_p2e/casewise_ptvreg_vs_mr_d_bscp_p2e.csv).
- Protocol and smoke/full-run checks: [execution specification](codex_execution_spec.md)
  and [technical log](technical_log.md).
