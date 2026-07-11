# Pareto Figure LR Validation

- Rows checked: `11`
- Blocking issues: `0`

| dataset | method_id | method_family | observed_lr_label | canonical_lr_label | status | selected_setting | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oasis20 | idir_original_dense_siren_be_autograd | idir | 1e-4 | 1e-4 | matches_standard | BE alpha sweep |  |
| oasis20 | sinr_cps2_be_bspline_analytic | sinr | 1e-4 | 1e-4 | matches_standard | BE alpha sweep |  |
| oasis20 | sinr_cps4_be_bspline_analytic | sinr | 1e-4 | 1e-4 | matches_standard | BE alpha sweep |  |
| oasis20 | direct_cp_cps2_be_bspline_analytic | directcp | 1e-3 | 1e-3 | matches_standard | BE alpha sweep |  |
| oasis20 | direct_cp_cps4_be_bspline_analytic | directcp | 1e-3 | 1e-3 | matches_standard | BE alpha sweep |  |
| oasis20 | direct_cp_8_4_2_be_bspline_analytic_multistage | directcp_multistage | 1e-2 | 1e-3 | temporary_exception | temporary multistage BE alpha sweep | Keep current OASIS multistage DirectCP curve in v001; replace after lr=1e-3 rerun. |
| lungct10 | idir_be_default_lungmask | idir | 1e-4 | 1e-4 | matches_standard | pTV-prep/native IDIR alpha sweep rows |  |
| lungct10 | sinr_cps4_ptv_lcc_isotv | sinr | 1e-4 | 1e-4 | matches_standard | single-stage pTV LCC+isoTV |  |
| lungct10 | direct_cp_cps4_ptv_lcc_isotv | directcp | 1e-3 | 1e-3 | matches_standard | single-stage pTV LCC+isoTV |  |
| lungct10 | direct_cp_32_16_8_4_ptv_lcc_isotv | directcp_multistage | 1e-3 | 1e-3 | matches_standard | 32-16-8-4 pTV LCC+isoTV sweep |  |
| lungct10 | direct_cp_32_16_8_4_global_ncc_fd_be | directcp_multistage | 1e-3 | 1e-3 | matches_standard | 32-16-8-4 global NCC+FD-BE sweep |  |
