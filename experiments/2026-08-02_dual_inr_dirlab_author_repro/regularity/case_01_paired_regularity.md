# DUAL-INR-DIR common regularity: case 01

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.0040920307 | 0.0040305323 | -6.1498402e-05 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.0041893625 | 0.004121105 | -6.8257417e-05 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.0043371259 | 0.0041786228 | -0.00015850306 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0025083169 | 0.0024531165 | -5.5200387e-05 |
| `log_jdet_std_full` | dimensionless | 0.098646755 | 0.097142603 | -0.0015041527 |
| `log_jdet_std_fg` | dimensionless | 0.09240053 | 0.092049739 | -0.0003507907 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.9662183 | 0.96559994 | -0.0006183613 |
| `jdet_mean_fg` | dimensionless | 0.92351981 | 0.92417961 | 0.00065979456 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 1.858254 | 1.8576077 | -0.00064633763 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 4.7391873 | 4.7165523 | -0.022635078 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 6.7291476 | 6.6477925 | -0.081355023 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 1.9972464 | 1.9959592 | -0.0012872196 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 4.579113 | 4.578589 | -0.00052400544 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 6.7291476 | 6.6477925 | -0.081355023 |
| `disp_mag_mean_full_mm` | mm | 3.4596246 | 3.4555657 | -0.0040588544 |
| `disp_mag_p95_full_mm` | mm | 9.9662893 | 9.8849923 | -0.081297039 |
| `disp_mag_max_full_mm` | mm | 13.310423 | 13.488771 | 0.17834772 |
| `disp_mag_mean_fg_mm` | mm | 3.8588116 | 3.8488932 | -0.0099183441 |
| `disp_mag_p95_fg_mm` | mm | 9.5835212 | 9.566977 | -0.016544203 |
| `disp_mag_max_fg_mm` | mm | 13.220221 | 13.236618 | 0.016397134 |

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
