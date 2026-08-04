# DUAL-INR-DIR common regularity: case 04

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.013186926 | 0.012745629 | -0.0004412969 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.011368862 | 0.011038051 | -0.00033081065 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.01002465 | 0.0099183425 | -0.00010630765 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0048743284 | 0.0047991863 | -7.514213e-05 |
| `log_jdet_std_full` | dimensionless | 0.18794524 | 0.18121763 | -0.0067276051 |
| `log_jdet_std_fg` | dimensionless | 0.18387664 | 0.18042101 | -0.0034556349 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.94895473 | 0.94906849 | 0.00011375654 |
| `jdet_mean_fg` | dimensionless | 0.87072234 | 0.87251759 | 0.0017952562 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 3.7981145 | 3.7426802 | -0.055434319 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 7.2875945 | 7.2321339 | -0.055460592 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 10.285062 | 10.280192 | -0.0048697079 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 4.504704 | 4.4811992 | -0.023504828 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 7.4362604 | 7.403332 | -0.032928404 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 9.6168945 | 9.6454367 | 0.028542158 |
| `disp_mag_mean_full_mm` | mm | 7.6757236 | 7.6009482 | -0.074775434 |
| `disp_mag_p95_full_mm` | mm | 16.363293 | 16.275268 | -0.088024913 |
| `disp_mag_max_full_mm` | mm | 21.349718 | 21.349761 | 4.3252639e-05 |
| `disp_mag_mean_fg_mm` | mm | 9.7855591 | 9.7366097 | -0.048949322 |
| `disp_mag_p95_fg_mm` | mm | 17.260666 | 17.229534 | -0.031131625 |
| `disp_mag_max_fg_mm` | mm | 20.273972 | 20.361095 | 0.087123299 |

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
