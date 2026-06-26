# LungCT Case08 DirectCP 4-Stage BE Implementation Comparison

## Summary

- Case: `08`.
- Method family: DirectCP 4-stage `32,16,8,4`.
- Shared settings: full-crop global NCC, alpha `10000`, downsample schedule
  `8,4,2,1`, stage epochs `200,400,800,1000`, LR `1e-3`, seed `1`.
- Compared regularizers:
  - dense forward-difference BE
  - analytic cubic B-spline BE
- Full jobs ran concurrently on GPUs `2,3`.

## Main Result

Original-snap TRE is exactly tied after snap-to-voxel rounding. Analytic
B-spline BE is marginally smoother and marginally better on subvoxel/1mm
diagnostic TREs, but took `16.1 s` longer (`+4.6%`) with the same peak memory.

| Regularizer | Original snap TRE | Original sub TRE | Reg1mm snap TRE | Reg1mm sub TRE | FG stdLogJ | FG J<=0 % | FG BE | FG diffusion | Time | Peak mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense FD-BE | 1.087 mm | 1.147 mm | 1.271 mm | 1.144 mm | 0.192354 | 0.0105 | 0.001444 | 0.013999 | 351.9 s | 6.49 GB |
| Analytic B-spline BE | 1.087 mm | 1.145 mm | 1.269 mm | 1.143 mm | 0.191389 | 0.0105 | 0.001333 | 0.013417 | 368.0 s | 6.49 GB |

## Differences

Analytic B-spline BE minus dense FD-BE:

```text
original_snap_tre_mean_mm     0.000000
original_subvoxel_tre_mean_mm -0.001392
reg1mm_snap_tre_mean_mm       -0.002002
reg1mm_subvoxel_tre_mean_mm   -0.001257
fg_logj_std                   -0.000965
fg_jdet_le0_pct               -0.000057
fg_bending_energy             -0.000112
fg_diffusion_l2               -0.000583
elapsed_sec                  +16.087421
peak_mem_gb                    0.000000
```

## Interpretation

For case `08` at alpha `10000`, the two BE implementations behave almost
identically in accuracy. The analytic B-spline BE has a small regularity
advantage by every foreground metric here, but the gain is tiny and does not
change the original-snap headline TRE.

The internal stage elapsed values are not used as the main timing comparison
because stage timing is not synchronized per stage. Use summary `elapsed_sec`
for the run-level optimization timing.

## Artifacts

- Output root:
  `/database/inrorspline/runs/lungct_case08_be_impl_compare_alpha1e4_v1`.
- Comparison rows: `comparison_rows.csv`, `comparison_rows.json`.
