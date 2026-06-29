# Automatic Result Selection Tables

- Generated UTC: `2026-06-26T22:17:07+00:00`
- OASIS20 selection maximizes mean Dice and reports first20 mean +/- standard deviation.
- LungCT10 selection minimizes original/native-crop snap TRE and keeps original subvoxel TRE as the compact diagnostic.
- LungCT pTVReg is a protocol reference row, not a clean method-only comparison.
- OASIS foreground regularity uses fixed raw image-positive foreground; LungCT foreground regularity uses fixed T00 lung mask on the pTV 1mm crop.

## OASIS seed-3 first20 automatic best-Dice summary

| Method | Selected | Cases | Dice | HD95 | ASD sym | FG stdLogJ | FG bending | FG Jdet<0 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDIR | alpha=1 | 20 | 0.775 +/- 0.039 | 2.36 +/- 0.73 | 0.53 +/- 0.16 | 0.74 +/- 0.11 | 0.03 +/- 0.00 | 2.27 +/- 0.88 |
| SINR cps2 | alpha=1000 | 20 | 0.800 +/- 0.038 | 2.27 +/- 0.64 | 0.46 +/- 0.14 | 0.94 +/- 0.14 | 0.10 +/- 0.01 | 5.41 +/- 1.39 |
| SINR cps4 | alpha=1000 | 20 | 0.796 +/- 0.040 | 2.32 +/- 0.73 | 0.48 +/- 0.16 | 0.95 +/- 0.14 | 0.10 +/- 0.01 | 5.68 +/- 1.48 |
| DirectCP cps2 | alpha=1000 | 20 | 0.803 +/- 0.040 | 2.37 +/- 0.80 | 0.47 +/- 0.18 | 0.93 +/- 0.16 | 0.14 +/- 0.03 | 5.84 +/- 1.46 |
| DirectCP cps4 | alpha=1000 | 20 | 0.798 +/- 0.040 | 2.32 +/- 0.72 | 0.47 +/- 0.16 | 0.94 +/- 0.14 | 0.12 +/- 0.02 | 5.81 +/- 1.57 |
| DirectCP 8-4-2 | alpha=1000 | 20 | 0.808 +/- 0.039 | 2.26 +/- 0.77 | 0.45 +/- 0.18 | 0.90 +/- 0.15 | 0.12 +/- 0.02 | 4.41 +/- 1.28 |
| Greedy | reference | 20 | 0.807 +/- 0.038 | 1.94 +/- 0.52 | 0.41 +/- 0.13 | 0.59 +/- 0.06 | 0.10 +/- 0.01 | 0.0061 +/- 0.0051 |
| VoxelMorph | reference | 20 | 0.772 +/- 0.044 | 2.53 +/- 0.76 | 0.55 +/- 0.20 | 0.74 +/- 0.05 | 0.25 +/- 0.02 | 3.24 +/- 0.73 |
| SITReg | reference | 20 | 0.824 +/- 0.027 | 1.75 +/- 0.43 | 0.35 +/- 0.09 | 0.58 +/- 0.07 | 0.39 +/- 0.04 | 0.0006 +/- 0.0005 |

## LungCT 4DCT all10 automatic best-TRE summary

| Method | Selected | Cases | Original snap TRE | Original sub TRE | FG stdLogJ | FG bending | FG Jdet<0 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IDIR | a=3 | 10 | 1.006 +/- 0.171 | 1.129 +/- 0.131 | 0.175 +/- 0.043 | 0.001202 +/- 0.000479 | 0.0089 +/- 0.0250 |
| DirectCP 4stage pTV-LCC+isoTV | w=0.11 | 10 | 0.922 +/- 0.152 | 1.049 +/- 0.119 | 0.180 +/- 0.050 | 0.002527 +/- 0.001038 | 0.0138 +/- 0.0193 |
| DirectCP 4stage global-NCC+FD-BE | a=10000 | 10 | 0.966 +/- 0.151 | 1.101 +/- 0.124 | 0.185 +/- 0.061 | 0.001971 +/- 0.001356 | 0.0037 +/- 0.0081 |
| SINR cps4 | default | 10 | 2.904 +/- 3.209 | 3.014 +/- 3.178 | 0.302 +/- 0.186 | 0.006339 +/- 0.006101 | 0.89 +/- 1.10 |
| DirectCP cps4 | default | 10 | 6.370 +/- 3.340 | 6.450 +/- 3.302 | 0.653 +/- 0.162 | 0.0341 +/- 0.0126 | 3.94 +/- 1.75 |
| pTVReg | pTVReg | 10 | 0.919 +/- 0.152 | 1.055 +/- 0.119 | 0.198 +/- 0.057 | 0.008956 +/- 0.003922 | 0.0278 +/- 0.0321 |

## LungCT single-stage failure cases

Failure is defined as `Original snap TRE` > `2.0` mm, using the same original/native-crop snap TRE convention as the LungCT headline table.

| Method | Failed cases | Fail count | Worst case |
| --- | --- | --- | --- |
| SINR cps4 | 05: 2.576, 06: 3.393, 07: 3.969, 08: 11.950, 10: 2.448 | 5/10 | 08: 11.950 |
| DirectCP cps4 | 02: 2.981, 03: 4.186, 04: 6.640, 05: 5.814, 06: 8.497, 07: 9.037, 08: 13.815, 09: 6.097, 10: 5.459 | 9/10 | 08: 13.815 |

## Generated Files

- `oasis20_summary.md`, `oasis20_summary.csv`, `oasis20_selected_rows.json`
- `lungct10_summary.md`, `lungct10_summary.csv`, `lungct10_selected_rows.json`
- `selection_manifest.json`

