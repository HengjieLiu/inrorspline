# D-BSCP a=40000 Case01/Case08 Screening Figure

This is a focused screening artifact, not a replacement for the v004 camera-ready source index.

- Figure: `population_overlay_plus_lungct_cases_2x4_tre_labels_camera_ready_fig3_d_bscp_a40000_case01_case08_screen.png`
- Summary CSV: `d_bscp_a40000_case01_case08_screen_summary.csv`
- Manifest: `d_bscp_a40000_case01_case08_screen_manifest.json`
- Changed only: DIR-LAB 4DCT representative case01/case08 Panel-A D-BSCP curves and TRE labels.
- Unchanged: OASIS, DIR-LAB population context, all Panel B curves, all non-D-BSCP methods.
- Baseline comparison uses the current v004 D-BSCP `a=50000` rows.

| Case | Baseline a=50000 TRE (mm) | Screen a=40000 TRE (mm) | Delta TRE (mm) | Baseline final 1-NCC | Screen final 1-NCC | Delta 1-NCC | Screen SDlogJ |
|---|---:|---:|---:|---:|---:|---:|---:|
| case01 | 1.073 | 1.065 | -0.008 | 0.03418 | 0.03324 | -0.00094 | 0.082 |
| case08 | 3.374 | 3.525 | 0.151 | 0.08292 | 0.07971 | -0.00321 | 0.341 |

Interpretation guardrail: the population columns are retained as v004 context. Because only case01 and case08 were rerun at `a=40000`, this figure should be used for alpha screening only.
