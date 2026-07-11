# COPD DirectCP64-32-16-8-4 pTV-LCC + isoTV E400x4+900 Sanity Run

Display name: `MR-D-BSCP 64-32-16-8-4 e400x4+900 (pTVReg loss)`.

Completed summaries: `10/10`.
Run root: `/database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity_v1`.

Headline metric: native-crop/original-domain snap-to-voxel TRE in mm.

## Mean Row

| Metric | Value |
| --- | --- |
| Orig snap TRE mean | 1.050 +/- 0.767 |
| Orig sub TRE mean | 1.157 +/- 0.737 |
| Reg1mm snap TRE mean | 1.291 +/- 0.730 |
| Reg1mm sub TRE mean | 1.153 +/- 0.738 |
| Mean SDlogJ | 0.4021 |
| Mean J<=0 % | 0.7251 |
| Mean elapsed sec | 382.9 |

## Per Case

| Case | Orig Snap TRE | Orig Sub TRE | Reg1mm Snap TRE | SDlogJ | J<=0 % | Elapsed s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.714 | 0.832 | 0.999 | 0.4862 | 1.9124 | 446.9 |
| 2 | 3.348 | 3.361 | 3.474 | 0.6158 | 2.0284 | 408.3 |
| 3 | 0.823 | 0.878 | 0.982 | 0.2743 | 0.0677 | 386.2 |
| 4 | 0.732 | 0.888 | 1.050 | 0.4401 | 0.4838 | 343.8 |
| 5 | 0.740 | 0.857 | 1.010 | 0.4078 | 0.3143 | 476.2 |
| 6 | 0.851 | 0.988 | 1.129 | 0.4873 | 1.3409 | 296.5 |
| 7 | 0.836 | 0.887 | 1.028 | 0.2579 | 0.0430 | 336.5 |
| 8 | 0.820 | 0.937 | 1.028 | 0.4177 | 0.5219 | 346.2 |
| 9 | 0.851 | 1.025 | 1.165 | 0.2618 | 0.2410 | 367.9 |
| 10 | 0.789 | 0.914 | 1.047 | 0.3723 | 0.2980 | 420.5 |

## Schedule Comparison vs P2d

Native-crop/original snap-to-voxel TRE in mm. Negative delta means the new `400,400,400,400,900` schedule improved over P2d `100,200,400,800,1000`.

| Case | P2d TRE | P2e TRE | Delta |
| --- | ---: | ---: | ---: |
| 01 | 1.303 | 0.714 | -0.589 |
| 02 | 3.335 | 3.348 | 0.013 |
| 03 | 0.823 | 0.823 | 0.000 |
| 04 | 7.261 | 0.732 | -6.528 |
| 05 | 5.873 | 0.740 | -5.132 |
| 06 | 0.986 | 0.851 | -0.135 |
| 07 | 0.849 | 0.836 | -0.013 |
| 08 | 3.164 | 0.820 | -2.345 |
| 09 | 0.639 | 0.851 | 0.212 |
| 10 | 0.793 | 0.789 | -0.005 |
| mean | 2.503 | 1.050 | -1.452 |

## Files

- `copd_case_rows.csv`
- `copd_mean_row.csv`
- `copd_case_rows.json`
- `copd_mean_row.json`
- `copd_schedule_comparison_vs_p2d.csv`
- `copd_schedule_comparison_vs_p2d.md`
