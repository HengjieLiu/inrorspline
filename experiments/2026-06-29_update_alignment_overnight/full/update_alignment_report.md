# Update-Alignment Overnight Report

Status: `completed`
Follow-up recommended: `False`

## Case Summary

| Dataset | Case | Method | Early alignment | Cumulative early alignment | Final metric |
|---|---|---|---:|---:|---:|
| oasis | oasis_rank009 | idir_original | 0.4817 | 0.9261 | 0.8110 dice_mean |
| oasis | oasis_rank009 | sinr_cps4_be_bspline_analytic | 0.3702 | 0.9006 | 0.8296 dice_mean |
| oasis | oasis_rank009 | direct_cp_cps4_be_bspline_analytic | 0.2789 | 0.6268 | 0.8268 dice_mean |
| lungct | lungct_case01 | idir_be_default_lungmask | 0.7939 | 0.9978 | 1.0626 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case01 | sinr_cps4_ptv_lcc_isotv | 0.8288 | 0.9990 | 1.0816 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case01 | direct_cp_cps4_ptv_lcc_isotv | 0.4633 | 0.8385 | 1.4658 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case01 | direct_cp_32_16_8_4_ptv_lcc_isotv | 0.8162 | 0.9857 | 1.0679 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case08 | idir_be_default_lungmask | 0.5487 | 0.9307 | 1.3306 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case08 | sinr_cps4_ptv_lcc_isotv | 0.2072 | 0.2697 | 12.1543 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case08 | direct_cp_cps4_ptv_lcc_isotv | 0.1152 | 0.2868 | 13.9884 reg1mm_snap_tre_mean_mm |
| lungct | lungct_case08 | direct_cp_32_16_8_4_ptv_lcc_isotv | 0.6549 | 0.9768 | 1.2130 reg1mm_snap_tre_mean_mm |

## SINR vs DirectCP Similarity

| Dataset | Case | Similarity |
|---|---|---:|
| lungct | lungct_case01 | 0.0009 |
| lungct | lungct_case08 | 0.2794 |
| oasis | oasis_rank009 | 0.3876 |

## Notes

- OASIS alignment uses fixed raw image-positive sampled foreground.
- LungCT alignment uses fixed T00 lung-mask samples on the canonical pTV 1mm crop.
- LungCT final registration metrics from the current runner are reg1mm crop TRE diagnostics; original-native pTVReg snap TRE is validated for the teacher field.
- These selected-case diagnostics are not a full population claim.
