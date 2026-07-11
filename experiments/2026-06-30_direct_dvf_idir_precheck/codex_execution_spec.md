# Direct Dense-DVF vs IDIR Precheck Execution Spec

Date: 2026-06-30

## Goal

Run a preliminary ablation that replaces IDIR's SIREN displacement field with a
directly optimized dense displacement grid while keeping the IDIR-style sampled
NCC loss and matched bending anchors.

## Methods

- OASIS: `direct_dvf_sampled_ncc_fd_be`
- LungCT: `direct_dvf_lungmask_sampled_ncc_fd_be`

Direct dense-grid DVF has no continuous analytic bending-energy formula. The
training regularizer is therefore dense finite-difference bending energy,
recorded as `dense_forward_difference_be`.

## Cases And Settings

| Dataset | Case | GPU | Alpha | LR | Epochs | Seed | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OASIS | rank9 `(0173,0136)` | 0 | 1 | 1e-4 | 2500 | 1 | 10000 |
| LungCT | case01 | 2 | alpha_bending 10 | 1e-4 | 2500 | 1 | 10000 |
| LungCT | case08 | 3 | alpha_bending 10 | 1e-4 | 2500 | 1 | 10000 |

GPU `1` is intentionally unused.

## Run Roots

- OASIS:
  `/database/inrorspline/runs/oasis_rank009_direct_dvf_idir_loss_precheck_v1`
- LungCT:
  `/database/inrorspline/runs/lungct_case01_case08_direct_dvf_idir_loss_precheck_v1`

## Preflight

1. Compile/import check for modified runners and helpers.
2. Validate OASIS rank9 resolves to original pair index `88`, fixed `0173`,
   moving `0136`.
3. Audit LungCT pTV-prep loader for cases `01,08`.
4. Run a two-epoch smoke for the new dense-DVF path before full launch.

## Aggregation Targets

Compare direct dense DVF against existing IDIR anchors:

- OASIS IDIR alpha `1`, LR `1e-4`.
- LungCT IDIR `alpha_bending=10`, LR `1e-4`.

Report accuracy/TRE, dense finite-difference BE, diffusion, log-Jacobian,
folding, NDV, runtime, and peak memory.
