# DUAL-INR-DIR common regularity: case 08

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.023369031 | 0.022872697 | -0.000496334 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.021793146 | 0.021507687 | -0.00028545897 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.0087547156 | 0.0085924047 | -0.00016231092 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0044750117 | 0.0044273258 | -4.7685862e-05 |
| `log_jdet_std_full` | dimensionless | 0.19199728 | 0.18376275 | -0.0082345249 |
| `log_jdet_std_fg` | dimensionless | 0.1552311 | 0.15091116 | -0.0043199325 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.91327565 | 0.9139955 | 0.00071985065 |
| `jdet_mean_fg` | dimensionless | 0.83472464 | 0.83515769 | 0.0004330532 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 8.6779327 | 8.647415 | -0.030517689 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 16.165996 | 15.986192 | -0.1798046 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 25.343521 | 25.347166 | 0.0036452495 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 8.0967579 | 8.0767301 | -0.020027746 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 14.692317 | 14.618407 | -0.073909941 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 25.343521 | 25.347166 | 0.0036452495 |
| `disp_mag_mean_full_mm` | mm | 14.307982 | 14.315484 | 0.0075018062 |
| `disp_mag_p95_full_mm` | mm | 28.519538 | 28.431583 | -0.087954866 |
| `disp_mag_max_full_mm` | mm | 33.631454 | 33.666161 | 0.034706627 |
| `disp_mag_mean_fg_mm` | mm | 14.61263 | 14.581238 | -0.031391909 |
| `disp_mag_p95_fg_mm` | mm | 28.268043 | 28.248413 | -0.019629824 |
| `disp_mag_max_fg_mm` | mm | 33.455647 | 33.440191 | -0.015455759 |

## Convention notes

- Finite-difference metrics in native crop voxel coordinates; they depend on voxel/component scaling and are not physical-coordinate energies.
- Dimensionless and invariant when converting voxel to physical coordinates because F_phys = S F_vox S^-1 is a similarity transform, so det(F_phys) = det(F_vox).
- Natural-log SD over positive determinants only.
- The shared implementation excludes a two-voxel border on axes longer than four voxels.
- Foreground: Exact fixed/T00 five-lobe union in the author crop.

## Posthoc source provenance

- Path: `${REPO_ROOT}/scripts/experiments/lungct_dual_inr_author/posthoc_author_regularity.py`
- Size: `20097` bytes
- SHA-256: `69883f85f50f0e7dc163a4064f2fdc9ab2257fb6981deb39bc87a5c93031fa8d`
