# LungCT FD-BE cps4 Alpha Extension Technical Log

## Known Limitations

- These runs extend only the unmasked full-crop global-NCC + dense FD-BE cps4 rows.
- Alpha values are not directly comparable to IDIR, pTVReg isoTV, or Dual-INR loss weights.
- The fixed T00 lung mask is metric-only here; no masked-loss optimization is being run.
- The source tree is expected to be dirty because this work is being developed alongside figure-refresh artifacts.

## Preflight Notes

- Dataset: pTVReg iso1mm DIR-Lab 4DCT, cases 01-10.
- Direction: T50 to T00.
- Methods:
  - `sinr_cps4_global_ncc_fd_be`, `lr=1e-4`, `alpha=100000,1000000`
  - `direct_cp_cps4_global_ncc_fd_be`, `lr=1e-3`, `alpha=100000,1000000`
- Acceptance target: 40/40 completed, finite TRE and foreground regularity metrics, 251 full-crop global-NCC diagnostic rows per job, no `masked_ncc` identifiers.

## Completion

- Dry-run: exactly 40 planned jobs.
- Smoke: 4/4 completed, one epoch per lane.
- Full launch: 40/40 completed with no launcher failures.
- Aggregation: accepted, 40/40 valid rows, zero validation errors.
- Figure refresh: v011 generated with OASIS rows matching v003 exactly and FD-BE cps4 alpha values `10000`, `100000`, and `1000000` for both INR-BSCP and D-BSCP.

## Mean Results

| Method | Alpha | TRE | SDlogJ | Folding % | Full 1-NCC | Fixed-mask 1-NCC |
|---|---:|---:|---:|---:|---:|---:|
| INR-BSCP cps4 FD-BE | 100000 | 1.043 | 0.1193 | 0.0000 | 0.00529 | 0.05004 |
| INR-BSCP cps4 FD-BE | 1000000 | 1.218 | 0.0858 | 0.0000 | 0.00817 | 0.07275 |
| D-BSCP cps4 FD-BE | 100000 | 1.016 | 0.1182 | 0.0000 | 0.00487 | 0.04621 |
| D-BSCP cps4 FD-BE | 1000000 | 1.209 | 0.0857 | 0.0000 | 0.00800 | 0.07153 |
