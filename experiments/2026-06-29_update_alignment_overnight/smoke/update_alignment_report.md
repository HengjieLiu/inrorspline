# Update-Alignment Overnight Report

Status: `completed`
Follow-up recommended: `False`

## Case Summary

| Dataset | Case | Method | Early alignment | Cumulative early alignment | Final metric |
|---|---|---|---:|---:|---:|
| oasis | oasis_rank009 | idir_original | 0.5325 | 0.8524 | 0.3972 dice_mean |
| oasis | oasis_rank009 | sinr_cps4_be_bspline_analytic | 0.6241 | 0.8672 | 0.5252 dice_mean |
| oasis | oasis_rank009 | direct_cp_cps4_be_bspline_analytic | 0.4309 | 0.4451 | 0.6397 dice_mean |
| lungct | lungct_case01 | idir_be_default_lungmask | 0.4938 | 0.8205 | 7.9770 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case01 | sinr_cps4_ptv_lcc_isotv | 0.6881 | 0.7700 | 7.9831 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case01 | direct_cp_cps4_ptv_lcc_isotv | 0.3424 | 0.3582 | 3.9274 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case01 | direct_cp_32_16_8_4_ptv_lcc_isotv | 0.6587 | 0.6982 | 3.6518 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case08 | idir_be_default_lungmask | 0.0816 | 0.3852 | 16.2129 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case08 | sinr_cps4_ptv_lcc_isotv | 0.2014 | 0.2368 | 15.1679 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case08 | direct_cp_cps4_ptv_lcc_isotv | 0.0465 | 0.0518 | 15.0837 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case08 | direct_cp_32_16_8_4_ptv_lcc_isotv | 0.2319 | 0.2524 | 14.9116 reg1mm_snap_tre_mean_mm |

## SINR vs DirectCP Similarity

| Dataset | Case | Similarity |
|---|---|---:|
| lungct | lungct_case01 | 0.2121 |
| lungct | lungct_case08 | -0.0291 |
| oasis | oasis_rank009 | -0.0160 |

## Notes

- OASIS alignment uses fixed raw image-positive sampled foreground.
- LungCT alignment uses fixed T00 lung-mask samples on the canonical pTV 1mm crop.
- LungCT final registration metrics from the current runner are reg1mm crop TRE diagnostics; original-native pTVReg snap TRE is validated for the teacher field.
- These selected-case diagnostics are not a full population claim.
