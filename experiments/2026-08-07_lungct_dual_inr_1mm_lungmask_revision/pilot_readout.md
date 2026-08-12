# Lung-Mask Revision Pilot Readout

This report covers the pTVReg-style 1 mm DUAL-INR branch using the fixed T00 lung mask as both optimization and metric mask.

| Variant | Loss branch | Native snap TRE | Reg-grid subvoxel TRE | FG SDlogJ | FG J<=0 pct |
|---|---|---:|---:|---:|---:|
| PL1D | `dual_native_reg` | n/a | n/a | n/a | n/a |
| PL1B | `idir_type_be` | n/a | n/a | n/a | n/a |
| PL1A | `dual_native_reg` | n/a | n/a | n/a | n/a |
| PL1C | `ptv_lcc_isotv` | 0.9909 +/- 0.1727 | 1.1011 +/- 0.1303 | 0.209918 +/- 0.059853 | 0.000119 +/- 0.000294 |

## Case Rows

| Variant | Case | Loss branch | Native snap TRE | Native subvoxel TRE | Sampling voxels | Metric voxels |
|---|---:|---|---:|---:|---:|---:|
| PL1C | 01 | `ptv_lcc_isotv` | 0.7703 | 0.9406 | 2318467 | 2318467 |
| PL1C | 02 | `ptv_lcc_isotv` | 0.7684 | 0.9356 | 5211915 | 5211915 |
| PL1C | 03 | `ptv_lcc_isotv` | 0.9104 | 1.0963 | 3955622 | 3955622 |
| PL1C | 04 | `ptv_lcc_isotv` | 1.3431 | 1.3608 | 2908142 | 2908142 |
| PL1C | 05 | `ptv_lcc_isotv` | 1.1406 | 1.2233 | 3273472 | 3273472 |
| PL1C | 06 | `ptv_lcc_isotv` | 0.9674 | 1.1028 | 2936192 | 2936192 |
| PL1C | 07 | `ptv_lcc_isotv` | 0.9469 | 1.0391 | 3550880 | 3550880 |
| PL1C | 08 | `ptv_lcc_isotv` | 1.1075 | 1.1981 | 5313087 | 5313087 |
| PL1C | 09 | `ptv_lcc_isotv` | 0.9965 | 1.0572 | 1893724 | 1893724 |
| PL1C | 10 | `ptv_lcc_isotv` | 0.9584 | 1.0316 | 3134549 | 3134549 |

Acceptance is technical only: expected artifacts, finite metrics, protocol identity, expected stages, and completed posthoc regularity. TRE and regularity magnitudes are decision inputs after inspection, not completion gates.
