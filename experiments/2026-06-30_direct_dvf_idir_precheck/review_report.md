# Direct Dense-DVF vs IDIR Precheck

Direct dense-DVF uses IDIR-style sampled NCC at 10,000 fixed-domain grid points.
Bending energy is dense finite difference; native continuous analytic BE is unavailable for dense-grid DVF.

## OASIS Rank9

| Case | Role | Dice | HD95 | ASD | BE fd fg vox | Diff fd fg vox | logJ std | fold % | NDV | sec | GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rank009 | IDIR anchor | 0.811 | 1.951 | 0.4112 | 0.02915 | 0.09352 | 0.6574 | 1.56 | 0.003699 | 125.7 | 4.123 |
| rank009 | direct dense DVF | 0.7673 | 2.69 | 0.5625 | 6.802 | 0.4937 | 1.258 | 9.889 | 0.08693 | 139.6 | 1.651 |

## LungCT pTV-Prep Diagnostic Domain

| Case | Role | snap TRE | subvox TRE | BE fg | Diff fg | logJ std | fold % | NDV | sec | GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| case01 | IDIR anchor | 1.063 | 0.9426 | 0.0003385 | 0.002921 | 0.09749 | 0 | 0 | 148.7 | 4.066 |
| case01 | direct dense DVF | 3.764 | 3.65 | 2.634 | 0.2558 | 1.028 | 4.514 | 0.04514 | 140.2 | 1.639 |
| case08 | IDIR anchor | 1.331 | 1.234 | 0.0005715 | 0.01207 | 0.173 | 0.0002471 | 2.471e-06 | 146.2 | 4.212 |
| case08 | direct dense DVF | 15.08 | 15.06 | 3.411 | 0.3164 | 1.211 | 5.233 | 0.05233 | 253.8 | 2.716 |

## Source Summaries

- `oasis_direct`: `/database/inrorspline/runs/oasis_rank009_direct_dvf_idir_loss_precheck_v1/direct_dvf_sampled_ncc_fd_be/direct_dvf_sampled_ncc_fd_be_rank009_a1_e2500_lr0p0001/summary.json`
- `oasis_idir`: `/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1/idir_original_dense_siren_be_autograd/idir_original_dense_siren_be_autograd_rank009_a1_e2500_lr0p0001/summary.json`
- `lung01_direct`: `/database/inrorspline/runs/lungct_case01_case08_direct_dvf_idir_loss_precheck_v1/direct_dvf_lungmask_sampled_ncc_fd_be/direct_dvf_lungmask_sampled_ncc_fd_be_case01_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung01_idir`: `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1/idir_be_default_lungmask/idir_be_default_lungmask_case01_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung08_direct`: `/database/inrorspline/runs/lungct_case01_case08_direct_dvf_idir_loss_precheck_v1/direct_dvf_lungmask_sampled_ncc_fd_be/direct_dvf_lungmask_sampled_ncc_fd_be_case08_e2500_lr0p0001_bend10_seed1/summary.json`
- `lung08_idir`: `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1/idir_be_default_lungmask/idir_be_default_lungmask_case08_e2500_lr0p0001_bend10_seed1/summary.json`
