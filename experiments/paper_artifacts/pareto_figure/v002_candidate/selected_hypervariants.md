# Pareto Figure Selected Hypervariants

| dataset | method_id | lr | variant |
| --- | --- | --- | --- |
| oasis20 | idir_original_dense_siren_be_autograd | 1e-4 | BE alpha sweep |
| oasis20 | sinr_cps2_be_bspline_analytic | 1e-4 | BE alpha sweep |
| oasis20 | sinr_cps4_be_bspline_analytic | 1e-4 | BE alpha sweep |
| oasis20 | direct_cp_cps2_be_bspline_analytic | 1e-3 | BE alpha sweep |
| oasis20 | direct_cp_cps4_be_bspline_analytic | 1e-3 | BE alpha sweep |
| oasis20 | direct_cp_8_4_2_be_bspline_analytic_multistage | 1e-3 | canonical lr=1e-3 multistage BE alpha sweep |
| lungct10 | idir_be_default_lungmask | 1e-4 | pTV-prep/native IDIR alpha sweep rows |
| lungct10 | sinr_cps4_ptv_lcc_isotv | 1e-4 | single-stage pTV LCC+isoTV |
| lungct10 | direct_cp_cps4_ptv_lcc_isotv | 1e-3 | single-stage pTV LCC+isoTV |
| lungct10 | direct_cp_32_16_8_4_ptv_lcc_isotv | 1e-3 | 32-16-8-4 pTV LCC+isoTV sweep |
| lungct10 | direct_cp_32_16_8_4_global_ncc_fd_be | 1e-3 | 32-16-8-4 global NCC+FD-BE sweep |
