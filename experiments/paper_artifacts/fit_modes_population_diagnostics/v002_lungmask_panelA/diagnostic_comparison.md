# v002 Lung-Mask Panel A Population Diagnostics

This is a no-overwrite candidate figure family. v001 remains the full-crop/schedule-matched historical comparison; v002 changes only DIR-LAB Panel A to fixed T00 lung-mask `1−NCC`.

## Main artifacts

- [dirlab_population_diagnostics_4x5.png](dirlab_population_diagnostics_4x5.png)
- [population_overlay_mean_std_2x2.png](population_overlay_mean_std_2x2.png)
- [population_overlay_plus_lungct_cases_2x4.png](population_overlay_plus_lungct_cases_2x4.png)
- [population_overlay_plus_lungct_cases_2x3.png](population_overlay_plus_lungct_cases_2x3.png)
- [population_overlay_plus_lungct_cases_2x4_tre_labels.png](population_overlay_plus_lungct_cases_2x4_tre_labels.png)

## Validation summary

- v002 source records: `160`.
- DIR-LAB Panel A lung-mask replacements: `40`.
- Unchanged non-DIR-LAB-Panel-A records: `120`.
- Axis definitions are locked to Panel A `1e-3–5e-1`, Panel B `1e-1–1e3`, epochs `0–2500`.
- Case01/case08 dual diagnostics are exact-compatible reused; cases02–07/09/10 are newly launched records when absent.
- MR-D-BSCP Panel A schedule is `400/400/800/900` for all DIR-LAB cases.

## Before/after endpoint comparison

| Case | Method | v001 full-crop 1−NCC | v002 lung-mask 1−NCC | Difference | TRE mean mm |
|---|---|---:|---:|---:|---:|
| case01 | INR-Dense | 0.0535893 | 0.031547 | -0.0220423 | 1.075 |
| case01 | INR-BSCP | 0.00224036 | 0.0318586 | 0.0296183 | 1.098 |
| case01 | D-BSCP | 0.00183827 | 0.0281935 | 0.0263553 | 1.073 |
| case01 | MR-D-BSCP | 0.00183976 | 0.0281414 | 0.0263016 | 1.068 |
| case02 | INR-Dense | 0.0497578 | 0.0274261 | -0.0223317 | 1.057 |
| case02 | INR-BSCP | 0.00156182 | 0.0227782 | 0.0212163 | 1.067 |
| case02 | D-BSCP | 0.00162202 | 0.0249678 | 0.0233458 | 1.186 |
| case02 | MR-D-BSCP | 0.00134057 | 0.0199022 | 0.0185617 | 1.070 |
| case03 | INR-Dense | 0.059229 | 0.0253545 | -0.0338745 | 1.198 |
| case03 | INR-BSCP | 0.00186306 | 0.0218263 | 0.0199633 | 1.195 |
| case03 | D-BSCP | 0.00227779 | 0.0284154 | 0.0261377 | 1.890 |
| case03 | MR-D-BSCP | 0.00160021 | 0.0190426 | 0.0174423 | 1.198 |
| case04 | INR-Dense | 0.131725 | 0.03153 | -0.100195 | 1.448 |
| case04 | INR-BSCP | 0.00380564 | 0.027007 | 0.0232014 | 1.462 |
| case04 | D-BSCP | 0.00464892 | 0.035242 | 0.0305931 | 2.449 |
| case04 | MR-D-BSCP | 0.0033794 | 0.0242578 | 0.0208784 | 1.450 |
| case05 | INR-Dense | 0.0727409 | 0.0370246 | -0.0357163 | 1.400 |
| case05 | INR-BSCP | 0.00234377 | 0.039369 | 0.0370252 | 2.126 |
| case05 | D-BSCP | 0.00279742 | 0.0498036 | 0.0470062 | 4.080 |
| case05 | MR-D-BSCP | 0.00182158 | 0.0298169 | 0.0279953 | 1.391 |
| case06 | INR-Dense | 0.222918 | 0.0771962 | -0.145722 | 1.262 |
| case06 | INR-BSCP | 0.00958693 | 0.0745324 | 0.0649455 | 1.605 |
| case06 | D-BSCP | 0.0128139 | 0.112202 | 0.0993885 | 6.599 |
| case06 | MR-D-BSCP | 0.00783145 | 0.0635362 | 0.0557047 | 1.219 |
| case07 | INR-Dense | 0.119458 | 0.0469806 | -0.0724776 | 1.164 |
| case07 | INR-BSCP | 0.00498646 | 0.0474055 | 0.0424191 | 2.006 |
| case07 | D-BSCP | 0.00844383 | 0.0865253 | 0.0780815 | 6.868 |
| case07 | MR-D-BSCP | 0.00392425 | 0.0365481 | 0.0326238 | 1.176 |
| case08 | INR-Dense | 0.0894971 | 0.0504403 | -0.0390569 | 1.339 |
| case08 | INR-BSCP | 0.00644851 | 0.0914712 | 0.0850227 | 7.046 |
| case08 | D-BSCP | 0.00795054 | 0.119902 | 0.111951 | 11.733 |
| case08 | MR-D-BSCP | 0.00276089 | 0.0362155 | 0.0334547 | 1.275 |
| case09 | INR-Dense | 0.148352 | 0.0398341 | -0.108518 | 1.191 |
| case09 | INR-BSCP | 0.00534725 | 0.039091 | 0.0337437 | 1.183 |
| case09 | D-BSCP | 0.0047121 | 0.0354837 | 0.0307716 | 1.160 |
| case09 | MR-D-BSCP | 0.00468922 | 0.035201 | 0.0305118 | 1.163 |
| case10 | INR-Dense | 0.0919531 | 0.0398991 | -0.052054 | 1.158 |
| case10 | INR-BSCP | 0.00330698 | 0.0350776 | 0.0317706 | 1.148 |
| case10 | D-BSCP | 0.00568807 | 0.0674388 | 0.0617507 | 4.283 |
| case10 | MR-D-BSCP | 0.00296468 | 0.0316025 | 0.0286378 | 1.120 |

## Notes

- Panel B is unchanged from v001 because the accepted supervised fitting curves already use the fixed-mask rRMSE convention.
- OASIS is regenerated/copied into v002 for side-by-side completeness but is intentionally unchanged.
