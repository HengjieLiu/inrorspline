# Best Result Table LR Validation

- Rows checked: `15`
- Blocking issues: `0`

| dataset | method_id | method_family | observed_lr_label | canonical_lr_label | status | selected_setting | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oasis20 | idir_original_dense_siren_be_autograd | idir | 1e-4 | 1e-4 | matches_standard | alpha=1 |  |
| oasis20 | sinr_cps2_be_bspline_analytic | sinr | 1e-4 | 1e-4 | matches_standard | alpha=1000 |  |
| oasis20 | sinr_cps4_be_bspline_analytic | sinr | 1e-4 | 1e-4 | matches_standard | alpha=1000 |  |
| oasis20 | direct_cp_cps2_be_bspline_analytic | directcp | 1e-3 | 1e-3 | matches_standard | alpha=1000 |  |
| oasis20 | direct_cp_cps4_be_bspline_analytic | directcp | 1e-3 | 1e-3 | matches_standard | alpha=1000 |  |
| oasis20 | direct_cp_8_4_2_be_bspline_analytic_multistage | directcp_multistage | 1e-3 | 1e-3 | matches_standard | alpha=1000 |  |
| oasis20 | greedy | reference |  |  | reference | reference |  |
| oasis20 | bs1_vxm_v0_2 | reference |  |  | reference | reference |  |
| oasis20 | bs4_sitreg_v0 | reference |  |  | reference | reference |  |
| lungct10 | idir_be_default_lungmask | idir | 1e-4 | 1e-4 | matches_standard | a=3 |  |
| lungct10 | direct_cp_32_16_8_4_ptv_lcc_isotv | directcp_multistage | 1e-3 | 1e-3 | matches_standard | w=0.11 |  |
| lungct10 | direct_cp_32_16_8_4_global_ncc_fd_be | directcp_multistage | 1e-3 | 1e-3 | matches_standard | a=10000 |  |
| lungct10 | sinr_cps4_ptv_lcc_isotv | sinr | 1e-4 | 1e-4 | matches_standard | default |  |
| lungct10 | direct_cp_cps4_ptv_lcc_isotv | directcp | 1e-3 | 1e-3 | matches_standard | default |  |
| lungct10 | ptvreg_reference | reference |  |  | reference | pTVReg |  |
