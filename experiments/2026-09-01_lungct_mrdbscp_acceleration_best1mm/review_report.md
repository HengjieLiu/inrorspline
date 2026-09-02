# LungCT MR-D-BSCP 1 mm acceleration and timing

- Experiment: `2026-09-01_lungct_mrdbscp_acceleration_best1mm`
- Case rows: `120/120`
- Timing endpoint: loaded input images to saved dense DVF (`loaded_images_to_dvf_file_sec`).
- Cold/warm timing: each variant/dataset worker uses case 1 as cold and cases 2–10 as warm; controls and optimizer state are reset per case.
- Queue time is excluded from per-case registration timing.
- Multistage optimization keeps per-epoch stage-grid expansion and disables non-final full-grid expansion.

## TRE aggregate

| Dataset | Variant | Cases | Steps mean +/- SD | Native snap TRE mm | Delta vs original | Reg-1mm snap TRE mm |
|---|---|---:|---:|---:|---:|---:|
| 4dct | original_corrected | 10 | 2500.0 +/- 0.0 | 0.917 +/- 0.158 | 0.000 | 1.164 +/- 0.121 |
| 4dct | frontloaded_fixed | 10 | 2600.0 +/- 0.0 | 0.918 +/- 0.156 | 0.001 | 1.163 +/- 0.122 |
| 4dct | oasis_bal_decay_cumulative | 10 | 625.0 +/- 95.7 | 0.918 +/- 0.155 | 0.000 | 1.165 +/- 0.123 |
| 4dct | oasis_fast_decay_cumulative | 10 | 505.0 +/- 61.0 | 0.921 +/- 0.157 | 0.004 | 1.165 +/- 0.122 |
| 4dct | oasis_bal_decay_ordinal | 10 | 632.5 +/- 70.8 | 0.921 +/- 0.157 | 0.003 | 1.164 +/- 0.121 |
| 4dct | oasis_fast_decay_ordinal | 10 | 492.5 +/- 45.7 | 0.920 +/- 0.156 | 0.002 | 1.164 +/- 0.121 |
| copd | original_corrected | 10 | 2500.0 +/- 0.0 | 1.044 +/- 0.741 | 0.000 | 1.274 +/- 0.690 |
| copd | frontloaded_fixed | 10 | 2600.0 +/- 0.0 | 1.032 +/- 0.692 | -0.013 | 1.264 +/- 0.651 |
| copd | oasis_bal_decay_cumulative | 10 | 1027.5 +/- 192.7 | 1.023 +/- 0.706 | -0.021 | 1.258 +/- 0.660 |
| copd | oasis_fast_decay_cumulative | 10 | 912.5 +/- 141.1 | 1.053 +/- 0.698 | 0.008 | 1.285 +/- 0.651 |
| copd | oasis_bal_decay_ordinal | 10 | 830.0 +/- 109.8 | 1.016 +/- 0.694 | -0.029 | 1.248 +/- 0.647 |
| copd | oasis_fast_decay_ordinal | 10 | 770.0 +/- 89.6 | 1.017 +/- 0.679 | -0.028 | 1.251 +/- 0.634 |

## Casewise comparison against fresh original 2500 baseline

Deltas are variant minus the same-dataset, same-case original rerun; negative timing delta means faster.

### 4dct

| Case | Variant | Actual steps | Native snap TRE mm | Delta TRE | Input -> DVF s | Delta timing s | Speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 01 | original_corrected | 2500 | 0.793 | 0.000 | 131.105 | 0.000 | 1.000 |
| 01 | frontloaded_fixed | 2600 | 0.793 | 0.000 | 130.796 | -0.308 | 1.002 |
| 01 | oasis_bal_decay_cumulative | 525 | 0.793 | 0.000 | 103.898 | -27.207 | 1.262 |
| 01 | oasis_fast_decay_cumulative | 450 | 0.793 | 0.000 | 114.725 | -16.380 | 1.143 |
| 01 | oasis_bal_decay_ordinal | 575 | 0.793 | 0.000 | 96.171 | -34.933 | 1.363 |
| 01 | oasis_fast_decay_ordinal | 450 | 0.792 | -0.001 | 117.973 | -13.132 | 1.111 |
| 02 | original_corrected | 2500 | 0.771 | 0.000 | 159.656 | 0.000 | 1.000 |
| 02 | frontloaded_fixed | 2600 | 0.780 | 0.009 | 148.150 | -11.506 | 1.078 |
| 02 | oasis_bal_decay_cumulative | 550 | 0.779 | 0.008 | 116.672 | -42.984 | 1.368 |
| 02 | oasis_fast_decay_cumulative | 450 | 0.779 | 0.008 | 129.481 | -30.176 | 1.233 |
| 02 | oasis_bal_decay_ordinal | 600 | 0.780 | 0.009 | 106.383 | -53.274 | 1.501 |
| 02 | oasis_fast_decay_ordinal | 450 | 0.780 | 0.009 | 135.854 | -23.802 | 1.175 |
| 03 | original_corrected | 2500 | 0.936 | 0.000 | 144.195 | 0.000 | 1.000 |
| 03 | frontloaded_fixed | 2600 | 0.948 | 0.012 | 135.267 | -8.928 | 1.066 |
| 03 | oasis_bal_decay_cumulative | 525 | 0.948 | 0.012 | 108.483 | -35.712 | 1.329 |
| 03 | oasis_fast_decay_cumulative | 450 | 0.957 | 0.020 | 118.845 | -25.351 | 1.213 |
| 03 | oasis_bal_decay_ordinal | 550 | 0.948 | 0.012 | 125.843 | -18.352 | 1.146 |
| 03 | oasis_fast_decay_ordinal | 450 | 0.945 | 0.008 | 124.760 | -19.435 | 1.156 |
| 04 | original_corrected | 2500 | 1.254 | 0.000 | 139.135 | 0.000 | 1.000 |
| 04 | frontloaded_fixed | 2600 | 1.246 | -0.008 | 136.537 | -2.598 | 1.019 |
| 04 | oasis_bal_decay_cumulative | 575 | 1.250 | -0.004 | 107.606 | -31.529 | 1.293 |
| 04 | oasis_fast_decay_cumulative | 475 | 1.254 | 0.000 | 114.961 | -24.173 | 1.210 |
| 04 | oasis_bal_decay_ordinal | 575 | 1.262 | 0.008 | 132.065 | -7.070 | 1.054 |
| 04 | oasis_fast_decay_ordinal | 475 | 1.250 | -0.004 | 125.809 | -13.326 | 1.106 |
| 05 | original_corrected | 2500 | 1.093 | 0.000 | 50.100 | 0.000 | 1.000 |
| 05 | frontloaded_fixed | 2600 | 1.090 | -0.004 | 38.790 | -11.309 | 1.292 |
| 05 | oasis_bal_decay_cumulative | 600 | 1.080 | -0.014 | 23.355 | -26.744 | 2.145 |
| 05 | oasis_fast_decay_cumulative | 450 | 1.091 | -0.003 | 23.912 | -26.188 | 2.095 |
| 05 | oasis_bal_decay_ordinal | 575 | 1.085 | -0.008 | 23.644 | -26.456 | 2.119 |
| 05 | oasis_fast_decay_ordinal | 450 | 1.091 | -0.003 | 21.921 | -28.178 | 2.285 |
| 06 | original_corrected | 2500 | 0.828 | 0.000 | 49.917 | 0.000 | 1.000 |
| 06 | frontloaded_fixed | 2600 | 0.830 | 0.002 | 41.068 | -8.849 | 1.215 |
| 06 | oasis_bal_decay_cumulative | 675 | 0.841 | 0.013 | 27.063 | -22.854 | 1.844 |
| 06 | oasis_fast_decay_cumulative | 525 | 0.833 | 0.005 | 22.070 | -27.847 | 2.262 |
| 06 | oasis_bal_decay_ordinal | 650 | 0.832 | 0.004 | 26.109 | -23.808 | 1.912 |
| 06 | oasis_fast_decay_ordinal | 525 | 0.836 | 0.008 | 22.419 | -27.498 | 2.227 |
| 07 | original_corrected | 2500 | 0.796 | 0.000 | 139.756 | 0.000 | 1.000 |
| 07 | frontloaded_fixed | 2600 | 0.789 | -0.007 | 133.351 | -6.405 | 1.048 |
| 07 | oasis_bal_decay_cumulative | 725 | 0.797 | 0.001 | 114.293 | -25.463 | 1.223 |
| 07 | oasis_fast_decay_cumulative | 600 | 0.801 | 0.004 | 108.927 | -30.829 | 1.283 |
| 07 | oasis_bal_decay_ordinal | 675 | 0.804 | 0.008 | 126.844 | -12.912 | 1.102 |
| 07 | oasis_fast_decay_ordinal | 550 | 0.794 | -0.002 | 122.945 | -16.811 | 1.137 |
| 08 | original_corrected | 2500 | 1.007 | 0.000 | 148.146 | 0.000 | 1.000 |
| 08 | frontloaded_fixed | 2600 | 1.009 | 0.002 | 143.953 | -4.193 | 1.029 |
| 08 | oasis_bal_decay_cumulative | 825 | 1.005 | -0.002 | 114.541 | -33.604 | 1.293 |
| 08 | oasis_fast_decay_cumulative | 600 | 1.008 | 0.000 | 125.994 | -22.152 | 1.176 |
| 08 | oasis_bal_decay_ordinal | 775 | 0.998 | -0.009 | 158.833 | 10.687 | 0.933 |
| 08 | oasis_fast_decay_ordinal | 575 | 1.008 | 0.001 | 136.302 | -11.844 | 1.087 |
| 09 | original_corrected | 2500 | 0.887 | 0.000 | 113.961 | 0.000 | 1.000 |
| 09 | frontloaded_fixed | 2600 | 0.885 | -0.002 | 114.891 | 0.931 | 0.992 |
| 09 | oasis_bal_decay_cumulative | 600 | 0.881 | -0.005 | 88.171 | -25.790 | 1.292 |
| 09 | oasis_fast_decay_cumulative | 500 | 0.895 | 0.008 | 98.042 | -15.918 | 1.162 |
| 09 | oasis_bal_decay_ordinal | 650 | 0.896 | 0.010 | 103.685 | -10.275 | 1.099 |
| 09 | oasis_fast_decay_ordinal | 500 | 0.893 | 0.006 | 102.308 | -11.652 | 1.114 |
| 10 | original_corrected | 2500 | 0.809 | 0.000 | 135.988 | 0.000 | 1.000 |
| 10 | frontloaded_fixed | 2600 | 0.809 | 0.000 | 128.353 | -7.635 | 1.059 |
| 10 | oasis_bal_decay_cumulative | 650 | 0.804 | -0.005 | 91.647 | -44.341 | 1.484 |
| 10 | oasis_fast_decay_cumulative | 550 | 0.801 | -0.008 | 108.539 | -27.449 | 1.253 |
| 10 | oasis_bal_decay_ordinal | 700 | 0.807 | -0.001 | 123.557 | -12.431 | 1.101 |
| 10 | oasis_fast_decay_ordinal | 500 | 0.807 | -0.001 | 121.076 | -14.912 | 1.123 |
### copd

| Case | Variant | Actual steps | Native snap TRE mm | Delta TRE | Input -> DVF s | Delta timing s | Speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 01 | original_corrected | 2500 | 0.736 | 0.000 | 196.947 | 0.000 | 1.000 |
| 01 | frontloaded_fixed | 2600 | 0.736 | 0.000 | 202.658 | 5.711 | 0.972 |
| 01 | oasis_bal_decay_cumulative | 1025 | 0.743 | 0.008 | 171.930 | -25.017 | 1.146 |
| 01 | oasis_fast_decay_cumulative | 925 | 0.731 | -0.005 | 193.301 | -3.646 | 1.019 |
| 01 | oasis_bal_decay_ordinal | 925 | 0.746 | 0.010 | 168.553 | -28.394 | 1.168 |
| 01 | oasis_fast_decay_ordinal | 800 | 0.749 | 0.014 | 164.853 | -32.093 | 1.195 |
| 02 | original_corrected | 2500 | 3.130 | 0.000 | 158.769 | 0.000 | 1.000 |
| 02 | frontloaded_fixed | 2600 | 2.985 | -0.145 | 165.050 | 6.281 | 0.962 |
| 02 | oasis_bal_decay_cumulative | 1200 | 3.014 | -0.116 | 140.606 | -18.163 | 1.129 |
| 02 | oasis_fast_decay_cumulative | 1050 | 2.997 | -0.133 | 156.961 | -1.807 | 1.012 |
| 02 | oasis_bal_decay_ordinal | 975 | 2.972 | -0.158 | 142.265 | -16.503 | 1.116 |
| 02 | oasis_fast_decay_ordinal | 850 | 2.931 | -0.199 | 138.633 | -20.136 | 1.145 |
| 03 | original_corrected | 2500 | 0.839 | 0.000 | 149.782 | 0.000 | 1.000 |
| 03 | frontloaded_fixed | 2600 | 0.841 | 0.002 | 150.903 | 1.121 | 0.993 |
| 03 | oasis_bal_decay_cumulative | 750 | 0.838 | -0.000 | 119.145 | -30.637 | 1.257 |
| 03 | oasis_fast_decay_cumulative | 675 | 0.838 | -0.000 | 138.613 | -11.169 | 1.081 |
| 03 | oasis_bal_decay_ordinal | 650 | 0.838 | -0.000 | 128.397 | -21.385 | 1.167 |
| 03 | oasis_fast_decay_ordinal | 675 | 0.826 | -0.012 | 127.050 | -22.732 | 1.179 |
| 04 | original_corrected | 2500 | 0.795 | 0.000 | 145.036 | 0.000 | 1.000 |
| 04 | frontloaded_fixed | 2600 | 0.797 | 0.002 | 155.378 | 10.342 | 0.933 |
| 04 | oasis_bal_decay_cumulative | 1100 | 0.735 | -0.060 | 120.012 | -25.024 | 1.209 |
| 04 | oasis_fast_decay_cumulative | 1025 | 0.788 | -0.007 | 132.582 | -12.454 | 1.094 |
| 04 | oasis_bal_decay_ordinal | 850 | 0.790 | -0.004 | 120.714 | -24.322 | 1.201 |
| 04 | oasis_fast_decay_ordinal | 825 | 0.790 | -0.005 | 123.773 | -21.263 | 1.172 |
| 05 | original_corrected | 2500 | 0.762 | 0.000 | 63.682 | 0.000 | 1.000 |
| 05 | frontloaded_fixed | 2600 | 0.793 | 0.032 | 52.105 | -11.578 | 1.222 |
| 05 | oasis_bal_decay_cumulative | 1350 | 0.798 | 0.036 | 33.613 | -30.069 | 1.895 |
| 05 | oasis_fast_decay_cumulative | 1025 | 0.805 | 0.043 | 33.602 | -30.081 | 1.895 |
| 05 | oasis_bal_decay_ordinal | 950 | 0.767 | 0.005 | 28.810 | -34.872 | 2.210 |
| 05 | oasis_fast_decay_ordinal | 900 | 0.775 | 0.013 | 28.207 | -35.476 | 2.258 |
| 06 | original_corrected | 2500 | 0.920 | 0.000 | 56.493 | 0.000 | 1.000 |
| 06 | frontloaded_fixed | 2600 | 0.908 | -0.012 | 47.890 | -8.603 | 1.180 |
| 06 | oasis_bal_decay_cumulative | 975 | 0.900 | -0.020 | 30.254 | -26.239 | 1.867 |
| 06 | oasis_fast_decay_cumulative | 950 | 1.154 | 0.234 | 30.559 | -25.934 | 1.849 |
| 06 | oasis_bal_decay_ordinal | 775 | 0.862 | -0.058 | 34.370 | -22.123 | 1.644 |
| 06 | oasis_fast_decay_ordinal | 750 | 0.920 | -0.000 | 31.121 | -25.371 | 1.815 |
| 07 | original_corrected | 2500 | 0.812 | 0.000 | 58.255 | 0.000 | 1.000 |
| 07 | frontloaded_fixed | 2600 | 0.797 | -0.014 | 46.041 | -12.214 | 1.265 |
| 07 | oasis_bal_decay_cumulative | 750 | 0.800 | -0.012 | 29.755 | -28.500 | 1.958 |
| 07 | oasis_fast_decay_cumulative | 700 | 0.795 | -0.017 | 23.917 | -34.338 | 2.436 |
| 07 | oasis_bal_decay_ordinal | 750 | 0.793 | -0.019 | 27.059 | -31.196 | 2.153 |
| 07 | oasis_fast_decay_ordinal | 650 | 0.777 | -0.035 | 26.637 | -31.618 | 2.187 |
| 08 | original_corrected | 2500 | 0.963 | 0.000 | 58.580 | 0.000 | 1.000 |
| 08 | frontloaded_fixed | 2600 | 0.941 | -0.022 | 45.561 | -13.019 | 1.286 |
| 08 | oasis_bal_decay_cumulative | 1200 | 0.979 | 0.015 | 31.875 | -26.705 | 1.838 |
| 08 | oasis_fast_decay_cumulative | 1050 | 0.983 | 0.020 | 32.599 | -25.981 | 1.797 |
| 08 | oasis_bal_decay_ordinal | 925 | 0.978 | 0.015 | 32.860 | -25.720 | 1.783 |
| 08 | oasis_fast_decay_ordinal | 850 | 0.968 | 0.005 | 34.533 | -24.048 | 1.696 |
| 09 | original_corrected | 2500 | 0.586 | 0.000 | 56.052 | 0.000 | 1.000 |
| 09 | frontloaded_fixed | 2600 | 0.628 | 0.042 | 48.346 | -7.706 | 1.159 |
| 09 | oasis_bal_decay_cumulative | 975 | 0.622 | 0.036 | 32.463 | -23.589 | 1.727 |
| 09 | oasis_fast_decay_cumulative | 800 | 0.637 | 0.051 | 28.351 | -27.701 | 1.977 |
| 09 | oasis_bal_decay_ordinal | 750 | 0.610 | 0.024 | 30.698 | -25.355 | 1.826 |
| 09 | oasis_fast_decay_ordinal | 650 | 0.628 | 0.042 | 29.207 | -26.845 | 1.919 |
| 10 | original_corrected | 2500 | 0.903 | 0.000 | 174.349 | 0.000 | 1.000 |
| 10 | frontloaded_fixed | 2600 | 0.891 | -0.012 | 163.984 | -10.365 | 1.063 |
| 10 | oasis_bal_decay_cumulative | 950 | 0.801 | -0.102 | 146.001 | -28.348 | 1.194 |
| 10 | oasis_fast_decay_cumulative | 925 | 0.799 | -0.104 | 175.159 | 0.810 | 0.995 |
| 10 | oasis_bal_decay_ordinal | 750 | 0.801 | -0.101 | 155.151 | -19.198 | 1.124 |
| 10 | oasis_fast_decay_ordinal | 750 | 0.802 | -0.101 | 154.575 | -19.774 | 1.128 |

## Per-case matrix and control-point geometry

Matrix sizes are recorded as Z x Y x X. Each stage entry gives the stage control-point spacing (`cps`), control-grid matrix, and total control-point count.

### 4dct

| Case | Matrix size Z x Y x X | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---:|---|---|---|---|---|
| 01 | 210x150x232 | cps=32; grid=11x9x12; control_points=1188 | cps=16; grid=18x14x20; control_points=5040 | cps=8; grid=32x24x36; control_points=27648 | cps=4; grid=60x44x68; control_points=179520 |
| 02 | 252x195x280 | cps=32; grid=12x11x13; control_points=1716 | cps=16; grid=20x18x22; control_points=7920 | cps=8; grid=36x32x40; control_points=46080 | cps=4; grid=68x60x76; control_points=310080 |
| 03 | 240x174x271 | cps=32; grid=12x10x13; control_points=1560 | cps=16; grid=20x16x22; control_points=7040 | cps=8; grid=36x28x40; control_points=40320 | cps=4; grid=68x52x76; control_points=268736 |
| 04 | 228x180x244 | cps=32; grid=12x10x12; control_points=1440 | cps=16; grid=20x16x20; control_points=6400 | cps=8; grid=36x28x36; control_points=36288 | cps=4; grid=68x52x68; control_points=240448 |
| 05 | 228x182x254 | cps=32; grid=12x10x12; control_points=1440 | cps=16; grid=20x16x20; control_points=6400 | cps=8; grid=36x28x36; control_points=36288 | cps=4; grid=68x52x68; control_points=240448 |
| 06 | 230x161x262 | cps=32; grid=12x10x13; control_points=1560 | cps=16; grid=20x16x22; control_points=7040 | cps=8; grid=36x28x40; control_points=40320 | cps=4; grid=68x52x76; control_points=268736 |
| 07 | 218x178x275 | cps=32; grid=11x10x13; control_points=1430 | cps=16; grid=18x16x22; control_points=6336 | cps=8; grid=32x28x40; control_points=35840 | cps=4; grid=60x52x76; control_points=237120 |
| 08 | 228x207x254 | cps=32; grid=12x11x12; control_points=1584 | cps=16; grid=20x18x20; control_points=7200 | cps=8; grid=36x32x36; control_points=41472 | cps=4; grid=68x60x68; control_points=277440 |
| 09 | 165x180x226 | cps=32; grid=10x10x12; control_points=1200 | cps=16; grid=16x16x20; control_points=5120 | cps=8; grid=28x28x36; control_points=28224 | cps=4; grid=52x52x68; control_points=183872 |
| 10 | 210x209x210 | cps=32; grid=11x11x11; control_points=1331 | cps=16; grid=18x18x18; control_points=5832 | cps=8; grid=32x32x32; control_points=32768 | cps=4; grid=60x60x60; control_points=216000 |
### copd

| Case | Matrix size Z x Y x X | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|---:|---|---|---|---|---|---|
| 01 | 275x201x229 | cps=64; grid=9x8x8; control_points=576 | cps=32; grid=14x12x12; control_points=2016 | cps=16; grid=24x20x20; control_points=9600 | cps=8; grid=44x36x36; control_points=57024 | cps=4; grid=84x68x68; control_points=388416 |
| 02 | 215x197x246 | cps=64; grid=8x8x8; control_points=512 | cps=32; grid=12x12x12; control_points=1728 | cps=16; grid=20x20x20; control_points=8000 | cps=8; grid=36x36x36; control_points=46656 | cps=4; grid=68x68x68; control_points=314432 |
| 03 | 252x188x264 | cps=64; grid=8x7x9; control_points=504 | cps=32; grid=12x10x14; control_points=1680 | cps=16; grid=20x16x24; control_points=7680 | cps=8; grid=36x28x44; control_points=44352 | cps=4; grid=68x52x84; control_points=297024 |
| 04 | 252x173x233 | cps=64; grid=8x7x8; control_points=448 | cps=32; grid=12x10x12; control_points=1440 | cps=16; grid=20x16x20; control_points=6400 | cps=8; grid=36x28x36; control_points=36288 | cps=4; grid=68x52x68; control_points=240448 |
| 05 | 262x193x248 | cps=64; grid=9x8x8; control_points=576 | cps=32; grid=14x12x12; control_points=2016 | cps=16; grid=24x20x20; control_points=9600 | cps=8; grid=44x36x36; control_points=57024 | cps=4; grid=84x68x68; control_points=388416 |
| 06 | 252x180x246 | cps=64; grid=8x7x8; control_points=448 | cps=32; grid=12x10x12; control_points=1440 | cps=16; grid=20x16x20; control_points=6400 | cps=8; grid=36x28x36; control_points=36288 | cps=4; grid=68x52x68; control_points=240448 |
| 07 | 255x167x251 | cps=64; grid=8x7x8; control_points=448 | cps=32; grid=12x10x12; control_points=1440 | cps=16; grid=20x16x20; control_points=6400 | cps=8; grid=36x28x36; control_points=36288 | cps=4; grid=68x52x68; control_points=240448 |
| 08 | 250x175x240 | cps=64; grid=8x7x8; control_points=448 | cps=32; grid=12x10x12; control_points=1440 | cps=16; grid=20x16x20; control_points=6400 | cps=8; grid=36x28x36; control_points=36288 | cps=4; grid=68x52x68; control_points=240448 |
| 09 | 240x191x264 | cps=64; grid=8x7x9; control_points=504 | cps=32; grid=12x10x14; control_points=1680 | cps=16; grid=20x16x24; control_points=7680 | cps=8; grid=36x28x44; control_points=44352 | cps=4; grid=68x52x84; control_points=297024 |
| 10 | 232x212x298 | cps=64; grid=8x8x9; control_points=576 | cps=32; grid=12x12x14; control_points=2016 | cps=16; grid=20x20x24; control_points=9600 | cps=8; grid=36x36x44; control_points=57024 | cps=4; grid=68x68x84; control_points=388416 |

## Lung-mask volumes

Volumes use the 1 mm isotropic masks: inspiration is the fixed mask and expiration is the moving mask. Labels are 4DCT fixed `T00` / moving `T50` and COPD fixed `iBH` / moving `eBH`. Percentage change is `(inspiration - expiration) / expiration * 100`.

### 4dct

| Case | Matrix size Z x Y x X | Inspiration mask voxels | Expiration mask voxels | Inspiration volume mL | Expiration volume mL | Difference mL | Percentage change vs expiration |
|---:|---|---:|---:|---:|---:|---:|---:|
| 01 | 210x150x232 | 2318467 | 2124109 | 2318.467 | 2124.109 | 194.358 | 9.15% |
| 02 | 252x195x280 | 5211915 | 4782517 | 5211.915 | 4782.517 | 429.398 | 8.98% |
| 03 | 240x174x271 | 3955622 | 3564469 | 3955.622 | 3564.469 | 391.153 | 10.97% |
| 04 | 228x180x244 | 2908142 | 2536690 | 2908.142 | 2536.690 | 371.452 | 14.64% |
| 05 | 228x182x254 | 3273472 | 2996300 | 3273.472 | 2996.300 | 277.172 | 9.25% |
| 06 | 230x161x262 | 2936192 | 2348426 | 2936.192 | 2348.426 | 587.766 | 25.03% |
| 07 | 218x178x275 | 3550880 | 2973649 | 3550.880 | 2973.649 | 577.231 | 19.41% |
| 08 | 228x207x254 | 5313087 | 4548701 | 5313.087 | 4548.701 | 764.386 | 16.80% |
| 09 | 165x180x226 | 1893724 | 1659003 | 1893.724 | 1659.003 | 234.721 | 14.15% |
| 10 | 210x209x210 | 3134549 | 2756169 | 3134.549 | 2756.169 | 378.380 | 13.73% |
### copd

| Case | Matrix size Z x Y x X | Inspiration mask voxels | Expiration mask voxels | Inspiration volume mL | Expiration volume mL | Difference mL | Percentage change vs expiration |
|---:|---|---:|---:|---:|---:|---:|---:|
| 01 | 275x201x229 | 6680780 | 4205086 | 6680.780 | 4205.086 | 2475.694 | 58.87% |
| 02 | 215x197x246 | 4028298 | 2992618 | 4028.298 | 2992.618 | 1035.680 | 34.61% |
| 03 | 252x188x264 | 5108177 | 4180748 | 5108.177 | 4180.748 | 927.429 | 22.18% |
| 04 | 252x173x233 | 4180503 | 2087473 | 4180.503 | 2087.473 | 2093.030 | 100.27% |
| 05 | 262x193x248 | 5030336 | 2516358 | 5030.336 | 2516.358 | 2513.978 | 99.91% |
| 06 | 252x180x246 | 4486693 | 2783877 | 4486.693 | 2783.877 | 1702.816 | 61.17% |
| 07 | 255x167x251 | 3802541 | 2869164 | 3802.541 | 2869.164 | 933.377 | 32.53% |
| 08 | 250x175x240 | 4672797 | 2742185 | 4672.797 | 2742.185 | 1930.612 | 70.40% |
| 09 | 240x191x264 | 3954376 | 2926544 | 3954.376 | 2926.544 | 1027.832 | 35.12% |
| 10 | 232x212x298 | 6968321 | 4305355 | 6968.321 | 4305.355 | 2662.966 | 61.85% |

## Timing aggregate

Primary timing is mean +/- sample SD; delta and speedup compare against the fresh original 2500 baseline for the same dataset and timing group.

| Dataset | Variant | Group | Cases | Input -> DVF s | Delta s | Speedup | Full case s | Optimizer s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 4dct | original_corrected | cold | 1 | 131.105 +/- 0.000 | 0.000 | 1.000 | 153.780 +/- 0.000 | 122.831 +/- 0.000 |
| 4dct | original_corrected | warm | 9 | 120.095 +/- 41.528 | 0.000 | 1.000 | 124.722 +/- 42.588 | 115.436 +/- 40.647 |
| 4dct | original_corrected | all_10 | 10 | 121.196 +/- 39.307 | 0.000 | 1.000 | 127.628 +/- 41.190 | 116.175 +/- 38.394 |
| 4dct | frontloaded_fixed | cold | 1 | 130.796 +/- 0.000 | -0.308 | 1.002 | 150.486 +/- 0.000 | 121.996 +/- 0.000 |
| 4dct | frontloaded_fixed | warm | 9 | 113.374 +/- 42.686 | -6.721 | 1.059 | 117.163 +/- 43.018 | 109.279 +/- 42.406 |
| 4dct | frontloaded_fixed | all_10 | 10 | 115.116 +/- 40.620 | -6.080 | 1.053 | 120.495 +/- 41.905 | 110.551 +/- 40.183 |
| 4dct | oasis_bal_decay_cumulative | cold | 1 | 103.898 +/- 0.000 | -27.207 | 1.262 | 125.607 +/- 0.000 | 93.960 +/- 0.000 |
| 4dct | oasis_bal_decay_cumulative | warm | 9 | 87.981 +/- 36.958 | -32.114 | 1.365 | 95.683 +/- 37.473 | 83.786 +/- 36.627 |
| 4dct | oasis_bal_decay_cumulative | all_10 | 10 | 89.573 +/- 35.206 | -31.623 | 1.353 | 98.675 +/- 36.575 | 84.804 +/- 34.681 |
| 4dct | oasis_fast_decay_cumulative | cold | 1 | 114.725 +/- 0.000 | -16.380 | 1.143 | 134.143 +/- 0.000 | 104.751 +/- 0.000 |
| 4dct | oasis_fast_decay_cumulative | warm | 9 | 94.530 +/- 41.643 | -25.565 | 1.270 | 99.831 +/- 42.300 | 89.783 +/- 41.452 |
| 4dct | oasis_fast_decay_cumulative | all_10 | 10 | 96.550 +/- 39.778 | -24.646 | 1.255 | 103.262 +/- 41.331 | 91.280 +/- 39.367 |
| 4dct | oasis_bal_decay_ordinal | cold | 1 | 96.171 +/- 0.000 | -34.933 | 1.363 | 123.997 +/- 0.000 | 87.002 +/- 0.000 |
| 4dct | oasis_bal_decay_ordinal | warm | 9 | 102.996 +/- 47.033 | -17.099 | 1.166 | 107.957 +/- 47.516 | 99.289 +/- 46.937 |
| 4dct | oasis_bal_decay_ordinal | all_10 | 10 | 102.313 +/- 44.395 | -18.882 | 1.185 | 109.561 +/- 45.085 | 98.060 +/- 44.423 |
| 4dct | oasis_fast_decay_ordinal | cold | 1 | 117.973 +/- 0.000 | -13.132 | 1.111 | 139.518 +/- 0.000 | 108.659 +/- 0.000 |
| 4dct | oasis_fast_decay_ordinal | warm | 9 | 101.488 +/- 46.035 | -18.606 | 1.183 | 106.191 +/- 47.290 | 97.430 +/- 46.493 |
| 4dct | oasis_fast_decay_ordinal | all_10 | 10 | 103.137 +/- 43.714 | -18.059 | 1.175 | 109.523 +/- 45.814 | 98.553 +/- 43.977 |
| copd | original_corrected | cold | 1 | 196.947 +/- 0.000 | 0.000 | 1.000 | 221.895 +/- 0.000 | 187.998 +/- 0.000 |
| copd | original_corrected | warm | 9 | 102.333 +/- 52.489 | 0.000 | 1.000 | 109.753 +/- 52.967 | 97.262 +/- 52.913 |
| copd | original_corrected | all_10 | 10 | 111.795 +/- 57.828 | 0.000 | 1.000 | 120.968 +/- 61.249 | 106.336 +/- 57.550 |
| copd | frontloaded_fixed | cold | 1 | 202.658 +/- 0.000 | 5.711 | 0.972 | 229.014 +/- 0.000 | 191.794 +/- 0.000 |
| copd | frontloaded_fixed | warm | 9 | 97.251 +/- 58.596 | -5.082 | 1.052 | 102.739 +/- 58.883 | 93.321 +/- 59.254 |
| copd | frontloaded_fixed | all_10 | 10 | 107.791 +/- 64.522 | -4.003 | 1.037 | 115.367 +/- 68.385 | 103.168 +/- 63.958 |
| copd | oasis_bal_decay_cumulative | cold | 1 | 171.930 +/- 0.000 | -25.017 | 1.146 | 200.277 +/- 0.000 | 160.733 +/- 0.000 |
| copd | oasis_bal_decay_cumulative | warm | 9 | 75.969 +/- 53.319 | -26.364 | 1.347 | 82.351 +/- 53.626 | 71.718 +/- 53.455 |
| copd | oasis_bal_decay_cumulative | all_10 | 10 | 85.565 +/- 58.718 | -26.229 | 1.307 | 94.144 +/- 62.824 | 80.620 +/- 57.726 |
| copd | oasis_fast_decay_cumulative | cold | 1 | 193.301 +/- 0.000 | -3.646 | 1.019 | 219.137 +/- 0.000 | 180.804 +/- 0.000 |
| copd | oasis_fast_decay_cumulative | warm | 9 | 83.594 +/- 64.922 | -18.739 | 1.224 | 89.067 +/- 66.290 | 79.275 +/- 65.452 |
| copd | oasis_fast_decay_cumulative | all_10 | 10 | 94.564 +/- 70.357 | -17.230 | 1.182 | 102.074 +/- 74.819 | 89.428 +/- 69.562 |
| copd | oasis_bal_decay_ordinal | cold | 1 | 168.553 +/- 0.000 | -28.394 | 1.168 | 192.330 +/- 0.000 | 157.627 +/- 0.000 |
| copd | oasis_bal_decay_ordinal | warm | 9 | 77.814 +/- 56.612 | -24.519 | 1.315 | 82.370 +/- 57.082 | 73.014 +/- 57.027 |
| copd | oasis_bal_decay_ordinal | all_10 | 10 | 86.888 +/- 60.599 | -24.907 | 1.287 | 93.366 +/- 64.073 | 81.475 +/- 60.055 |
| copd | oasis_fast_decay_ordinal | cold | 1 | 164.853 +/- 0.000 | -32.093 | 1.195 | 193.853 +/- 0.000 | 153.603 +/- 0.000 |
| copd | oasis_fast_decay_ordinal | warm | 9 | 77.082 +/- 56.589 | -25.251 | 1.328 | 81.947 +/- 57.237 | 72.752 +/- 57.029 |
| copd | oasis_fast_decay_ordinal | all_10 | 10 | 85.859 +/- 60.141 | -25.936 | 1.302 | 93.137 +/- 64.532 | 80.837 +/- 59.536 |

## Stage and materialization validation

- Stage rows written: `540`; detailed table: `stage_rows.csv`.
- Case trace counts: `{'[false, false, false, true]': 60, '[false, false, false, false, true]': 60}`.
- Case stop-reason payloads: `82` distinct; exact stage reasons are in `stage_rows.csv`.
- Expected final-only traces: 4DCT `[false,false,false,true]`; COPD `[false,false,false,false,true]`.

## Validation

- Validation errors: `0`
- All 120 rows are complete, finite, and satisfy the final-only materialization trace.
