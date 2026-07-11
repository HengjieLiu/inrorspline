# LungCT IDIR Sampled pTV-LCC + TV 0.11 Pilot

Output root: `/database/inrorspline/runs/lungct_idir_sampled_ptvlcc_tv_w0p11_all10_v1`

Aggregation errors: `[]`

## Mean Row

| Method | Cases | Original Snap TRE | Original Sub TRE | 1mm Snap TRE | 1mm Sub TRE | FG logJ std | FG J<=0% | BE fg | Diff fg | Time sec | Peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDIR sampled pTV-LCC + TV | 10 | 1.400 | 1.476 | 1.579 | 1.483 | 0.171860 | 0.0477 | 0.001448 | 0.010796 | 31.7 | 0.58 |

## Per-Case Rows

| Case | Original Identity TRE | Original Snap TRE | Original Sub TRE | 1mm Snap TRE | 1mm Sub TRE | FG logJ std | FG J<=0% | Time sec | Peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 3.892 | 0.792 | 0.956 | 1.075 | 0.960 | 0.097407 | 0.0000 | 32.1 | 0.50 |
| 02 | 4.338 | 0.857 | 0.998 | 1.109 | 0.998 | 0.079826 | 0.0000 | 29.4 | 0.69 |
| 03 | 6.943 | 1.087 | 1.166 | 1.258 | 1.165 | 0.101859 | 0.0000 | 31.8 | 0.62 |
| 04 | 9.830 | 1.504 | 1.492 | 1.573 | 1.482 | 0.195974 | 0.0000 | 29.2 | 0.58 |
| 05 | 7.477 | 1.397 | 1.404 | 1.503 | 1.407 | 0.211012 | 0.0125 | 33.0 | 0.59 |
| 06 | 10.891 | 1.227 | 1.323 | 1.410 | 1.324 | 0.192093 | 0.0000 | 31.5 | 0.57 |
| 07 | 11.026 | 1.141 | 1.240 | 1.329 | 1.248 | 0.181568 | 0.0000 | 31.6 | 0.60 |
| 08 | 14.995 | 3.895 | 3.892 | 4.026 | 3.945 | 0.286352 | 0.4542 | 33.7 | 0.64 |
| 09 | 7.918 | 1.058 | 1.151 | 1.245 | 1.152 | 0.175566 | 0.0000 | 33.8 | 0.48 |
| 10 | 7.301 | 1.043 | 1.141 | 1.265 | 1.147 | 0.196942 | 0.0100 | 30.8 | 0.56 |

## Notes

- This is an IDIR-specific sampled approximation to pTV-LCC+isoTV, not the dense pTVReg/control-grid objective.
- Training samples pTV 1mm border-crop coordinates; fixed T00 lung mask is used only for foreground metrics.
- Original Snap TRE is the pTVReg-comparable native/crop snap-to-voxel convention.
