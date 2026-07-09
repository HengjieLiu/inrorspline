# Completed First100 Baselines And MR-D-BSCP Metrics

Values are population means over 100 OASIS seed-3 cases. Full mean/std columns are in `first100_completed_baselines_mrd_bscp_metrics.csv`.

## Accuracy

| Method | Alpha | Dice up | HD95 down | ASD warped-to-fixed down | ASD symmetric down |
|---|---:|---:|---:|---:|---:|
| BS1 VoxelMorph | - | 0.768 | 2.540 | 0.600 | 0.560 |
| BS2 TransMorph | - | 0.795 | 2.149 | 0.494 | 0.455 |
| BS3 VFA | - | 0.822 | 1.885 | 0.404 | 0.371 |
| BS4 SITReg | - | 0.825 | 1.816 | 0.374 | 0.356 |
| Greedy | - | 0.809 | 2.018 | 0.436 | 0.411 |
| MR-D-BSCP 8-4-2 | 0 | 0.759 | 3.411 | 0.770 | 0.636 |
| MR-D-BSCP 8-4-2 | 10 | 0.789 | 2.831 | 0.604 | 0.521 |
| MR-D-BSCP 8-4-2 | 100 | 0.805 | 2.426 | 0.510 | 0.456 |
| MR-D-BSCP 8-4-2 | 1000 | 0.810 | 2.168 | 0.457 | 0.427 |
| MR-D-BSCP 8-4-2 | 10000 | 0.798 | 2.105 | 0.465 | 0.450 |
| MR-D-BSCP 8-4-2 | 100000 | 0.763 | 2.298 | 0.558 | 0.553 |

## Regularity

| Method | Alpha | BE FD fg down | Diff FD fg down | Diff L2 fg down | J<=0 % fg down | std(logJ) fg down | NDV frac fg down |
|---|---:|---:|---:|---:|---:|---:|---:|
| BS1 VoxelMorph | - | 0.247 | 0.110 | 0.110 | 3.019 | 0.733 | 0.013 |
| BS2 TransMorph | - | 0.154 | 0.095 | 0.095 | 2.227 | 0.688 | 0.008 |
| BS3 VFA | - | 0.483 | 0.122 | 0.122 | 0.189 | 0.613 | 6.55e-04 |
| BS4 SITReg | - | 0.396 | 0.124 | 0.124 | 5.02e-04 | 0.566 | 2.60e-04 |
| Greedy | - | 0.103 | 0.092 | 0.092 | 0.006 | 0.591 | 6.62e-06 |
| MR-D-BSCP 8-4-2 | 0 | 3.382 | 1.026 | 1.026 | 30.023 | 1.421 | 0.507 |
| MR-D-BSCP 8-4-2 | 10 | 1.061 | 0.492 | 0.492 | 16.543 | 1.267 | 0.142 |
| MR-D-BSCP 8-4-2 | 100 | 0.348 | 0.275 | 0.275 | 8.438 | 1.076 | 0.045 |
| MR-D-BSCP 8-4-2 | 1000 | 0.102 | 0.157 | 0.157 | 3.862 | 0.872 | 0.013 |
| MR-D-BSCP 8-4-2 | 10000 | 0.023 | 0.078 | 0.078 | 1.016 | 0.642 | 0.002 |
| MR-D-BSCP 8-4-2 | 100000 | 0.003 | 0.028 | 0.028 | 0.065 | 0.368 | 3.81e-05 |
