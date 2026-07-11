# COPD DirectCP64-32-16-8-4 pTV-LCC + isoTV Sanity Run

Display name: `MR-D-BSCP 64-32-16-8-4 (pTVReg loss)`.

Completed summaries: `10/10`.
Run root: `/database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity_v1`.

Headline metric: native-crop/original-domain snap-to-voxel TRE in mm.

## Mean Row

| Metric | Value |
| --- | --- |
| Orig snap TRE mean | 2.503 +/- 2.254 |
| Orig sub TRE mean | 2.593 +/- 2.240 |
| Reg1mm snap TRE mean | 2.718 +/- 2.228 |
| Reg1mm sub TRE mean | 2.598 +/- 2.251 |
| Mean SDlogJ | 0.4778 |
| Mean J<=0 % | 2.1468 |
| Mean elapsed sec | 369.5 |

## Per Case

| Case | Orig Snap TRE | Orig Sub TRE | Reg1mm Snap TRE | SDlogJ | J<=0 % | Elapsed s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.303 | 1.418 | 1.584 | 0.5115 | 2.1941 | 432.7 |
| 2 | 3.335 | 3.342 | 3.444 | 0.6196 | 1.9909 | 385.6 |
| 3 | 0.823 | 0.878 | 0.980 | 0.2750 | 0.0635 | 354.7 |
| 4 | 7.261 | 7.327 | 7.436 | 0.7802 | 6.6890 | 341.6 |
| 5 | 5.873 | 5.971 | 6.078 | 0.7207 | 6.2647 | 451.1 |
| 6 | 0.986 | 1.124 | 1.261 | 0.4906 | 1.4412 | 333.1 |
| 7 | 0.849 | 0.887 | 1.028 | 0.2575 | 0.0417 | 338.3 |
| 8 | 3.164 | 3.258 | 3.364 | 0.5116 | 2.3053 | 317.1 |
| 9 | 0.639 | 0.814 | 0.957 | 0.2376 | 0.1662 | 355.5 |
| 10 | 0.793 | 0.914 | 1.052 | 0.3734 | 0.3113 | 385.1 |

## Motion Range vs Post TRE

Native-crop/original-domain TRE in mm. Pre-registration TRE is the fixed-to-moving landmark distance before registration; post TRE is the completed 64-32-16-8-4 run.

| Case | N | Pre mean | Pre P90 | Pre P95 | Pre max | Post snap mean | Post sub mean | Mean reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 300 | 26.334 | 39.074 | 44.309 | 56.073 | 1.303 | 1.418 | 25.031 |
| 02 | 300 | 21.786 | 30.110 | 31.491 | 40.390 | 3.335 | 3.342 | 18.451 |
| 03 | 300 | 12.639 | 22.565 | 23.720 | 28.931 | 0.823 | 0.878 | 11.816 |
| 04 | 300 | 29.584 | 46.709 | 52.101 | 66.500 | 7.261 | 7.327 | 22.323 |
| 05 | 300 | 30.083 | 49.681 | 52.634 | 62.513 | 5.873 | 5.971 | 24.210 |
| 06 | 300 | 28.456 | 41.578 | 46.994 | 52.176 | 0.986 | 1.124 | 27.470 |
| 07 | 300 | 21.602 | 33.182 | 35.677 | 39.151 | 0.849 | 0.887 | 20.752 |
| 08 | 300 | 26.457 | 46.378 | 51.231 | 60.441 | 3.164 | 3.258 | 23.292 |
| 09 | 300 | 14.860 | 32.077 | 36.420 | 42.964 | 0.639 | 0.814 | 14.221 |
| 10 | 300 | 21.806 | 36.951 | 39.904 | 46.991 | 0.793 | 0.914 | 21.013 |
| case_mean | 3000 | 23.361 | 37.831 | 41.448 | 49.613 | 2.503 | 2.593 | 20.858 |
| pooled_all_landmarks | 3000 | 23.361 | 39.034 | 44.989 | 66.500 | 2.503 | 2.593 | 20.858 |

## Files

- `copd_case_rows.csv`
- `copd_mean_row.csv`
- `copd_case_rows.json`
- `copd_mean_row.json`
- `copd_motion_range_vs_post_tre.csv`
- `copd_motion_range_vs_post_tre.md`
