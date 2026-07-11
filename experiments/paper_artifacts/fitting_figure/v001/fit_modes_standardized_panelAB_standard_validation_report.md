# Merged Panel A/B LR Validation

- Rows checked: `24`
- Blocking issues: `0`

| dataset | method_id | method_family | observed_lr_label | canonical_lr_label | status | selected_setting | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OASIS rank-9 | idir_original_dense_siren_be_autograd | idir | 1e-4 | 1e-4 | matches_standard | alpha=1, diagnostic full-image global NCC every 10 epochs | Existing matched-eval run; diagnostic only, logged every 10 epochs. |
| OASIS rank-9 | sinr_cps4_be_bspline_analytic | sinr | 1e-4 | 1e-4 | matches_standard | alpha=1000, optimized full-image global NCC + BE | Existing OASIS lr-sweep history; plotted data loss is the optimized image term. |
| OASIS rank-9 | direct_cp_cps4_be_bspline_analytic | directcp | 1e-3 | 1e-3 | matches_standard | alpha=1000, optimized full-image global NCC + BE | Existing OASIS lr-sweep history; plotted data loss is the optimized image term. |
| OASIS rank-9 | direct_cp_8_4_2_be_bspline_analytic_multistage | directcp_multistage | 1e-3 | 1e-3 | matches_standard | alpha=1000, cps 8/4/2, downsample 4/2/1, stage epochs 400/800/1300, diagnostic full-image global NCC every 10 epochs | New OASIS multistage run: cps 8/4/2, downsample 4/2/1, stage epochs 400/800/1300, alpha 1000. |
| LungCT case01 | idir_be_default_lungmask | idir | 1e-4 | 1e-4 | matches_standard | alpha_bending=3, optimized native IDIR sampled NCC + BE, diagnostic full-crop global NCC every 10 epochs | New IDIR rerun with full-crop global NCC diagnostic every 10 epochs; alpha_bending 3. |
| LungCT case01 | sinr_cps4_global_ncc_fd_be | sinr | 1e-4 | 1e-4 | matches_standard | alpha=10000, optimized full-crop global NCC + dense FD-BE | New SINR cps4 global-NCC+FD-BE run; alpha 10000. |
| LungCT case01 | direct_cp_cps4_global_ncc_fd_be | directcp | 1e-3 | 1e-3 | matches_standard | alpha=10000, optimized full-crop global NCC + dense FD-BE | New DirectCP cps4 global-NCC+FD-BE run; alpha 10000. |
| LungCT case01 | direct_cp_32_16_8_4_global_ncc_fd_be | directcp_multistage | 1e-3 | 1e-3 | matches_standard | alpha=10000, cps 32/16/8/4, downsample 8/4/2/1, stage epochs 200/400/800/1000 | New DirectCP multistage global-NCC+FD-BE run: cps 32/16/8/4, downsample 8/4/2/1, stage epochs 200/400/800/1000, alpha 10000. |
| LungCT case08 | idir_be_default_lungmask | idir | 1e-4 | 1e-4 | matches_standard | alpha_bending=3, optimized native IDIR sampled NCC + BE, diagnostic full-crop global NCC every 10 epochs | New IDIR rerun with full-crop global NCC diagnostic every 10 epochs; alpha_bending 3. |
| LungCT case08 | sinr_cps4_global_ncc_fd_be | sinr | 1e-4 | 1e-4 | matches_standard | alpha=10000, optimized full-crop global NCC + dense FD-BE | New SINR cps4 global-NCC+FD-BE run; alpha 10000. |
| LungCT case08 | direct_cp_cps4_global_ncc_fd_be | directcp | 1e-3 | 1e-3 | matches_standard | alpha=10000, optimized full-crop global NCC + dense FD-BE | New DirectCP cps4 global-NCC+FD-BE run; alpha 10000. |
| LungCT case08 | direct_cp_32_16_8_4_global_ncc_fd_be | directcp_multistage | 1e-3 | 1e-3 | matches_standard | alpha=10000, cps 32/16/8/4, downsample 8/4/2/1, stage epochs 200/400/800/1000 | New DirectCP multistage global-NCC+FD-BE run: cps 32/16/8/4, downsample 8/4/2/1, stage epochs 200/400/800/1000, alpha 10000. |
| OASIS SITReg | IDIR | idir | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE in the OASIS field-fit sweep. |
| OASIS SITReg | SINR cps4 | sinr | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE in the OASIS field-fit sweep. |
| OASIS SITReg | DirectCP cps4 | directcp | 1e-3 | 1e-3 | matches_standard | 1e-3 | Requested LR trajectory for masked normalized-grid target-field MSE in the OASIS field-fit sweep. |
| OASIS SITReg | DirectCP 8-4-2 | directcp_multistage | 1e-3 | 1e-3 | matches_standard | 1e-3 | Canonical OASIS multistage DirectCP field-fit add-on: cps 8/4/2, downsample 4/2/1, stage epochs 400/800/1300, lr 1e-3; plotted using finest-domain masked normalized-grid MSE diagnostics. |
| LungCT case01 | IDIR | idir | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case01 | SINR cps4 | sinr | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case01 | DirectCP cps4 | directcp | 1e-3 | 1e-3 | matches_standard | 1e-3 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case01 | DirectCP 4stage cps4 | directcp_multistage | 1e-3 | 1e-3 | matches_standard | 1e-3 | Requested LR trajectory using posthoc finest-domain masked normalized-grid MSE. Coarse stages log this diagnostic sparsely; the final 1mm stage logs every epoch. |
| LungCT case08 | IDIR | idir | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case08 | SINR cps4 | sinr | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case08 | DirectCP cps4 | directcp | 1e-3 | 1e-3 | matches_standard | 1e-3 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case08 | DirectCP 4stage cps4 | directcp_multistage | 1e-3 | 1e-3 | matches_standard | 1e-3 | Requested LR trajectory using posthoc finest-domain masked normalized-grid MSE. Coarse stages log this diagnostic sparsely; the final 1mm stage logs every epoch. |
