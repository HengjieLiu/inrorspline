# LungCT FD-BE cps4 alpha-landscape aggregation

- Updated: `2026-08-15 03:36:35 +0000`
- Scope: unmasked full-crop global NCC + dense FD-BE, LungCT cases 01-10.
- New landscape alphas in this run: `1e4, 2e4, 5e4, 2e5, 5e5`.
- Fixed T00 lung mask is metric-only in these rows.
- Acceptance: `True` (`100/100` valid rows).

| Method | alpha | Cases | TRE | SDlogJ | Folding % | Full 1-NCC | Fixed-mask 1-NCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| INR-BSCP cps4 (FD-BE) | 10000 | 10 | 1.762 | 0.2360 | 0.2308 | 0.00415 | 0.04304 |
| D-BSCP cps4 (FD-BE) | 10000 | 10 | 3.894 | 0.4265 | 1.4257 | 0.00528 | 0.05882 |
| INR-BSCP cps4 (FD-BE) | 20000 | 10 | 1.474 | 0.1795 | 0.0659 | 0.00431 | 0.04402 |
| D-BSCP cps4 (FD-BE) | 20000 | 10 | 2.743 | 0.2854 | 0.4920 | 0.00491 | 0.05324 |
| INR-BSCP cps4 (FD-BE) | 50000 | 10 | 1.042 | 0.1351 | 0.0000 | 0.00470 | 0.04496 |
| D-BSCP cps4 (FD-BE) | 50000 | 10 | 1.306 | 0.1592 | 0.0749 | 0.00458 | 0.04643 |
| INR-BSCP cps4 (FD-BE) | 200000 | 10 | 1.069 | 0.1068 | 0.0000 | 0.00582 | 0.05385 |
| D-BSCP cps4 (FD-BE) | 200000 | 10 | 1.057 | 0.1065 | 0.0000 | 0.00558 | 0.05212 |
| INR-BSCP cps4 (FD-BE) | 500000 | 10 | 1.156 | 0.0940 | 0.0000 | 0.00705 | 0.06406 |
| D-BSCP cps4 (FD-BE) | 500000 | 10 | 1.144 | 0.0940 | 0.0000 | 0.00681 | 0.06204 |
