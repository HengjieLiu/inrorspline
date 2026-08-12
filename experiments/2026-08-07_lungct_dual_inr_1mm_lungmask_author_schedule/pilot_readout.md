# Lung-Mask Revision Pilot Readout

This report covers the pTVReg-style 1 mm DUAL-INR branch using the fixed T00 lung mask as both optimization and metric mask.

| Variant | Loss branch | Native snap TRE | Reg-grid subvoxel TRE | FG SDlogJ | FG J<=0 pct |
|---|---|---:|---:|---:|---:|
| PL1D | `dual_native_reg` | n/a | n/a | n/a | n/a |
| PL1B | `idir_type_be` | n/a | n/a | n/a | n/a |
| PL1A | `dual_native_reg` | 0.9758 +/- 0.1752 | 1.1144 +/- 0.1280 | 0.143408 +/- 0.037411 | 0.000000 +/- 0.000000 |

## Case Rows

| Variant | Case | Loss branch | Native snap TRE | Native subvoxel TRE | Sampling voxels | Metric voxels |
|---|---:|---|---:|---:|---:|---:|
| PL1A | 01 | `dual_native_reg` | 0.7529 | 0.9496 | 2318467 | 2318467 |
| PL1A | 02 | `dual_native_reg` | 0.7079 | 0.9309 | 5211915 | 5211915 |
| PL1A | 03 | `dual_native_reg` | 0.8963 | 1.0842 | 3955622 | 3955622 |
| PL1A | 04 | `dual_native_reg` | 1.2893 | 1.3323 | 2908142 | 2908142 |
| PL1A | 05 | `dual_native_reg` | 1.1842 | 1.2923 | 3273472 | 3273472 |
| PL1A | 06 | `dual_native_reg` | 0.9748 | 1.1300 | 2936192 | 2936192 |
| PL1A | 07 | `dual_native_reg` | 0.9432 | 1.0837 | 3550880 | 3550880 |
| PL1A | 08 | `dual_native_reg` | 1.0424 | 1.1582 | 5313087 | 5313087 |
| PL1A | 09 | `dual_native_reg` | 1.0067 | 1.0859 | 1893724 | 1893724 |
| PL1A | 10 | `dual_native_reg` | 0.9602 | 1.0591 | 3134549 | 3134549 |

Acceptance is technical only: expected artifacts, finite metrics, protocol identity, expected stages, and completed posthoc regularity. TRE and regularity magnitudes are decision inputs after inspection, not completion gates.
