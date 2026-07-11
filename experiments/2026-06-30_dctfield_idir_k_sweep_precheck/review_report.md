# DCTField IDIR-Loss K-Sweep Precheck

DCTField optimizes low-frequency cosine coefficients and decodes them to dense displacement fields.
Training uses IDIR-style sampled NCC and dense finite-difference BE on the decoded field.

## OASIS Rank9

| Case | Role | K | Dice | HD95 | ASD | BE | Diff | logJ | fold % | NDV | sec | GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rank009 | IDIR anchor |  | 0.811 | 1.951 | 0.4112 | 0.02915 | 0.09352 | 0.6574 | 1.56 | 0.003699 | 125.7 | 4.123 |
| rank009 | direct dense DVF |  | 0.7673 | 2.69 | 0.5625 | 6.802 | 0.4937 | 1.258 | 9.889 | 0.08693 | 139.6 | 1.651 |
| rank009 | DCTField | 16 | 0.6738 | 3.485 | 0.8451 | 0.0007475 | 0.003231 | 0.1131 | 0 | 0 | 138 | 1.435 |
| rank009 | DCTField | 32 | 0.7242 | 2.574 | 0.6676 | 0.03621 | 0.04019 | 0.4842 | 0.2653 | 0.0004184 | 140.3 | 1.437 |
| rank009 | DCTField | 64 | 0.7624 | 2.649 | 0.5953 | 0.5584 | 0.1854 | 1.019 | 11 | 0.05957 | 140.7 | 1.447 |
| rank009 | DCTField | 128 | 0.7781 | 2.494 | 0.5541 | 4.052 | 0.4616 | 1.187 | 16.89 | 0.1924 | 142 | 1.529 |

## LungCT pTV-Prep Diagnostic Domain

| Case | Role | K | snap TRE | subvox TRE | BE | Diff | logJ | fold % | NDV | sec | GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case01 | IDIR anchor |  | 1.063 | 0.9426 | 0.0003385 | 0.002921 | 0.09749 | 0 | 0 | 148.7 | 4.066 |
| case01 | direct dense DVF |  | 3.764 | 3.65 | 2.634 | 0.2558 | 1.028 | 4.514 | 0.04514 | 140.2 | 1.639 |
| case01 | DCTField | 16 | 3.826 | 3.771 | 0.0004164 | 0.001901 | 0.09575 | 0 | 0 | 137.8 | 1.409 |
| case01 | DCTField | 32 | 3.652 | 3.61 | 0.01974 | 0.02236 | 0.4126 | 0.739 | 0.00739 | 138.7 | 1.41 |
| case01 | DCTField | 64 | 3.662 | 3.595 | 0.2996 | 0.0969 | 0.6906 | 4.4 | 0.044 | 139.6 | 1.42 |
| case01 | DCTField | 128 | 3.913 | 3.748 | 1.934 | 0.2297 | 0.8659 | 6.92 | 0.0692 | 141.5 | 1.503 |
| case08 | IDIR anchor |  | 1.331 | 1.234 | 0.0005715 | 0.01207 | 0.173 | 0.0002471 | 2.471e-06 | 146.2 | 4.212 |
| case08 | direct dense DVF |  | 15.08 | 15.06 | 3.411 | 0.3164 | 1.211 | 5.233 | 0.05233 | 253.8 | 2.716 |
| case08 | DCTField | 16 | 15.08 | 15.06 | 0.0001261 | 0.0008053 | 0.06404 | 0 | 0 | 248.1 | 2.329 |
| case08 | DCTField | 32 | 15.02 | 14.99 | 0.006829 | 0.01105 | 0.2724 | 0.1467 | 0.001467 | 249.3 | 2.329 |
| case08 | DCTField | 64 | 14.99 | 15 | 0.1368 | 0.06293 | 0.6496 | 3.451 | 0.03451 | 255.8 | 2.34 |
| case08 | DCTField | 128 | 15.08 | 15.05 | 1.117 | 0.1708 | 0.8421 | 6.522 | 0.06522 | 254.3 | 2.422 |

## Interpretation

- OASIS best DCT K by Dice: `128` with Dice `0.7781` and foreground folding `16.89%` (direct dense DVF folding `9.889%`).
- LungCT case01 best DCT K by snap TRE: `32` with TRE `3.652` mm and folding `0.739%` (direct dense DVF TRE `3.764` mm, folding `4.514%`).
- LungCT case08 best DCT K by snap TRE: `64` with TRE `14.99` mm and folding `3.451%` (direct dense DVF TRE `15.08` mm, folding `5.233%`).

## Source Summaries

- `oasis_idir`: `/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1/idir_original_dense_siren_be_autograd/idir_original_dense_siren_be_autograd_rank009_a1_e2500_lr0p0001/summary.json`
- `oasis_direct`: `/database/inrorspline/runs/oasis_rank009_direct_dvf_idir_loss_precheck_v1/direct_dvf_sampled_ncc_fd_be/direct_dvf_sampled_ncc_fd_be_rank009_a1_e2500_lr0p0001/summary.json`
- `lung01_idir`: `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1/idir_be_default_lungmask/idir_be_default_lungmask_case01_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung01_direct`: `/database/inrorspline/runs/lungct_case01_case08_direct_dvf_idir_loss_precheck_v1/direct_dvf_lungmask_sampled_ncc_fd_be/direct_dvf_lungmask_sampled_ncc_fd_be_case01_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung08_idir`: `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1/idir_be_default_lungmask/idir_be_default_lungmask_case08_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung08_direct`: `/database/inrorspline/runs/lungct_case01_case08_direct_dvf_idir_loss_precheck_v1/direct_dvf_lungmask_sampled_ncc_fd_be/direct_dvf_lungmask_sampled_ncc_fd_be_case08_e2500_lr0p0001_bend10_seed1/summary.json`
- `oasis_dct_k16`: `/database/inrorspline/runs/oasis_rank009_dctfield_idir_k_sweep_precheck_v1/dctfield_sampled_ncc_fd_be/dctfield_sampled_ncc_fd_be_rank009_k16_a1_e2500_lr0p0001/summary.json`
- `lung01_dct_k16`: `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1/dctfield_lungmask_sampled_ncc_fd_be/dctfield_lungmask_sampled_ncc_fd_be_case01_k16_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung08_dct_k16`: `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1/dctfield_lungmask_sampled_ncc_fd_be/dctfield_lungmask_sampled_ncc_fd_be_case08_k16_e2500_lr0p0001_bend10_seed1/summary.json`
- `oasis_dct_k32`: `/database/inrorspline/runs/oasis_rank009_dctfield_idir_k_sweep_precheck_v1/dctfield_sampled_ncc_fd_be/dctfield_sampled_ncc_fd_be_rank009_k32_a1_e2500_lr0p0001/summary.json`
- `lung01_dct_k32`: `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1/dctfield_lungmask_sampled_ncc_fd_be/dctfield_lungmask_sampled_ncc_fd_be_case01_k32_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung08_dct_k32`: `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1/dctfield_lungmask_sampled_ncc_fd_be/dctfield_lungmask_sampled_ncc_fd_be_case08_k32_e2500_lr0p0001_bend10_seed1/summary.json`
- `oasis_dct_k64`: `/database/inrorspline/runs/oasis_rank009_dctfield_idir_k_sweep_precheck_v1/dctfield_sampled_ncc_fd_be/dctfield_sampled_ncc_fd_be_rank009_k64_a1_e2500_lr0p0001/summary.json`
- `lung01_dct_k64`: `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1/dctfield_lungmask_sampled_ncc_fd_be/dctfield_lungmask_sampled_ncc_fd_be_case01_k64_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung08_dct_k64`: `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1/dctfield_lungmask_sampled_ncc_fd_be/dctfield_lungmask_sampled_ncc_fd_be_case08_k64_e2500_lr0p0001_bend10_seed1/summary.json`
- `oasis_dct_k128`: `/database/inrorspline/runs/oasis_rank009_dctfield_idir_k_sweep_precheck_v1/dctfield_sampled_ncc_fd_be/dctfield_sampled_ncc_fd_be_rank009_k128_a1_e2500_lr0p0001/summary.json`
- `lung01_dct_k128`: `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1/dctfield_lungmask_sampled_ncc_fd_be/dctfield_lungmask_sampled_ncc_fd_be_case01_k128_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung08_dct_k128`: `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1/dctfield_lungmask_sampled_ncc_fd_be/dctfield_lungmask_sampled_ncc_fd_be_case08_k128_e2500_lr0p0001_bend10_seed1/summary.json`
