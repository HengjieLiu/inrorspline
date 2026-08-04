# DUAL-INR-DIR Author A0/A1 Aggregate

Cohort: `production`; cases: `01,02,03,04,05,06,07,08,09,10`; seed: `42`.
All requested A0/A1 records passed the DONE/artifact/history/provenance gates.

## TRE

| Group | Dense snap TRE (mm) | Dense subvoxel TRE (mm) | Direct INR snap TRE (mm) |
|---|---:|---:|---:|
| A0 | 0.9789 ± 0.1727 | 1.1084 ± 0.1319 | 0.9789 ± 0.1727 |
| A1 | 0.9834 ± 0.1669 | 1.1078 ± 0.1316 | 0.9834 ± 0.1669 |
| A1_minus_A0 | 0.0045 ± 0.0139 | -0.0006 ± 0.0033 | 0.0045 ± 0.0139 |

Values are across-case mean ± sample SD; paired deltas are `A1 - A0`.

## Training components

| Group | Final total | Final similarity | Final spatial | Final mask |
|---|---:|---:|---:|---:|
| A0 | 0.0524 ± 0.0195 | 0.0445 ± 0.0177 | 0.0060 ± 0.0019 | 0.0019 ± 0.0005 |
| A1 | 0.0503 ± 0.0191 | 0.0445 ± 0.0176 | 0.0059 ± 0.0019 | 0.0000 ± 0.0000 |
| A1_minus_A0 | -0.0021 ± 0.0006 | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 | -0.0019 ± 0.0005 |

## Runtime and peak memory

| Group | Training seconds | Total seconds | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|
| A0 | 358.9300 ± 16.9269 | 422.1443 ± 56.6434 | 7.5666 ± 0.0757 | 9.1104 ± 0.4285 |
| A1 | 350.2951 ± 15.7771 | 404.7574 ± 35.3217 | 7.5666 ± 0.0757 | 9.1104 ± 0.4285 |
| A1_minus_A0 | -8.6349 ± 17.4974 | -17.3869 ± 40.9275 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

## Schedule replay audit

| Case | A0 boundaries | A1 boundaries | Exact match |
|---:|---|---|---|
| 01 | `[0, 332, 564]` | `[0, 332, 564]` | yes |
| 02 | `[0, 278, 559]` | `[0, 278, 559]` | yes |
| 03 | `[0, 297, 468]` | `[0, 297, 468]` | yes |
| 04 | `[0, 286, 455]` | `[0, 286, 455]` | yes |
| 05 | `[0, 359, 574]` | `[0, 359, 574]` | yes |
| 06 | `[0, 478, 661]` | `[0, 478, 661]` | yes |
| 07 | `[0, 372, 618]` | `[0, 372, 618]` | yes |
| 08 | `[0, 308, 482]` | `[0, 308, 482]` | yes |
| 09 | `[0, 362, 524]` | `[0, 362, 524]` | yes |
| 10 | `[0, 370, 665]` | `[0, 370, 665]` | yes |

## Common posthoc regularity

Status: **pass**; computed by the separate validated posthoc gate.

| Group | Diffusion FG | Bending FG | SDlogJ FG | J<=0 FG (%) |
|---|---:|---:|---:|---:|
| A0 | 0.0130 ± 0.0071 | 0.0046 ± 0.0020 | 0.1401 ± 0.0361 | 0.0000 ± 0.0000 |
| A1 | 0.0128 ± 0.0070 | 0.0045 ± 0.0020 | 0.1389 ± 0.0353 | 0.0000 ± 0.0000 |
| A1_minus_A0 | -0.0002 ± 0.0001 | -0.0001 ± 0.0000 | -0.0012 ± 0.0015 | 0.0000 ± 0.0000 |

- Scope: Shared finite-difference metrics on author crop voxel fields, with full-crop and fixed/T00 five-lobe-union scopes; displacement magnitude additionally reported in mm.
- Unit caveat: Diffusion and bending are voxel-coordinate finite-difference metrics, not physical-coordinate energies; Jacobian metrics are dimensionless.
- Source hash: `b44325a46804d2bf8c0ad8af2b62d72b890f459a107aedd7b9f7ce9666ea56b5`.


This report is descriptive. It does not establish equivalence or a method-only comparison with IDIR/pTVReg.
