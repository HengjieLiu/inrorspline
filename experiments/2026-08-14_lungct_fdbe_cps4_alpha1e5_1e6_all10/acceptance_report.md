# LungCT FD-BE cps4 alpha-extension aggregation

- Updated: `2026-08-15 00:17:56 +0000`
- Scope: unmasked full-crop global NCC + dense FD-BE, LungCT cases 01-10.
- Fixed T00 lung mask is metric-only in these rows.
- Acceptance: `True` (`40/40` valid rows).

| Method | alpha | Cases | TRE | SDlogJ | Folding % | Full 1-NCC | Fixed-mask 1-NCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| INR-BSCP cps4 (FD-BE) | 100000 | 10 | 1.043 | 0.1193 | 0.0000 | 0.00529 | 0.05004 |
| INR-BSCP cps4 (FD-BE) | 1e+06 | 10 | 1.218 | 0.0858 | 0.0000 | 0.00817 | 0.07275 |
| D-BSCP cps4 (FD-BE) | 100000 | 10 | 1.016 | 0.1182 | 0.0000 | 0.00487 | 0.04621 |
| D-BSCP cps4 (FD-BE) | 1e+06 | 10 | 1.209 | 0.0857 | 0.0000 | 0.00800 | 0.07153 |
