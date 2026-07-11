# DCTField IDIR-Loss K-Sweep Precheck Execution Spec

Date: 2026-06-30

## Goal

Run a 4-size DCT-compressed displacement-field precheck on the same three
diagnostic cases used by the direct dense-DVF precheck. The aim is to compare a
non-neural, non-spline compressed deformation space against IDIR and direct
dense-field optimization.

## Methods

- OASIS: `dctfield_sampled_ncc_fd_be`
- LungCT: `dctfield_lungmask_sampled_ncc_fd_be`

DCTField optimizes coefficients `[3,K,K,K]`, decodes them to a dense normalized
grid displacement field with an orthonormal separable cosine basis, and trains
with sampled NCC plus dense finite-difference BE on the decoded field.

## Cases And Settings

| Dataset | Case | GPU queue | K values | Alpha | LR | Epochs | Seed | Samples |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| OASIS | rank9 `(0173,0136)` | 0 | 16,32,64,128 | 1 | 1e-4 | 2500 | 1 | 10000 |
| LungCT | case01 | 2 | 16,32,64,128 | alpha_bending 10 | 1e-4 | 2500 | 1 | 10000 |
| LungCT | case08 | 3 | 16,32,64,128 | alpha_bending 10 | 1e-4 | 2500 | 1 | 10000 |

GPU `1` is intentionally unused.

## Run Roots

- OASIS:
  `/database/inrorspline/runs/oasis_rank009_dctfield_idir_k_sweep_precheck_v1`
- LungCT:
  `/database/inrorspline/runs/lungct_case01_case08_dctfield_idir_k_sweep_precheck_v1`

## Preflight

1. Compile/import check for modified runners, shared DCT helper, sanity script,
   and aggregation script.
2. DCT decoder sanity check: zero decode, shape, gradients, and overlarge-K
   failure.
3. Validate OASIS rank9 resolves to original pair index `88`, fixed `0173`,
   moving `0136`.
4. Audit LungCT pTV-prep loader for cases `01,08`.
5. Run two-epoch K128 smokes for OASIS rank9 and LungCT case08.

## Aggregation Targets

Compare DCTField K `16,32,64,128` against:

- OASIS IDIR alpha `1`, LR `1e-4`.
- LungCT IDIR `alpha_bending=10`, LR `1e-4`.
- Completed direct dense-DVF precheck P7.

Report accuracy/TRE, dense finite-difference BE, diffusion, log-Jacobian,
folding, NDV, final loss terms, runtime, and peak memory.
