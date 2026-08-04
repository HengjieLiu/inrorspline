# DUAL-INR-DIR common regularity: case 05

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.012091736 | 0.011867226 | -0.00022450985 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.012434804 | 0.012263553 | -0.00017125006 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.0083998195 | 0.0083354041 | -6.4415421e-05 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0048285655 | 0.0047403408 | -8.8224703e-05 |
| `log_jdet_std_full` | dimensionless | 0.14502866 | 0.14377055 | -0.0012581057 |
| `log_jdet_std_fg` | dimensionless | 0.14902268 | 0.14783049 | -0.00119219 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.95574526 | 0.95672929 | 0.00098403008 |
| `jdet_mean_fg` | dimensionless | 0.90841818 | 0.90909647 | 0.00067828877 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 3.5221139 | 3.4914379 | -0.030676015 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 8.060192 | 8.0136068 | -0.046585193 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 10.698022 | 10.722053 | 0.024030375 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 3.7296391 | 3.7222973 | -0.0073418025 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 7.7288132 | 7.7162466 | -0.01256661 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 10.470606 | 10.463763 | -0.0068428827 |
| `disp_mag_mean_full_mm` | mm | 6.8518947 | 6.8231302 | -0.028764469 |
| `disp_mag_p95_full_mm` | mm | 18.052584 | 17.923828 | -0.12875575 |
| `disp_mag_max_full_mm` | mm | 23.988976 | 23.591283 | -0.39769324 |
| `disp_mag_mean_fg_mm` | mm | 7.495939 | 7.482342 | -0.013597018 |
| `disp_mag_p95_fg_mm` | mm | 18.009497 | 18.005105 | -0.0043926744 |
| `disp_mag_max_fg_mm` | mm | 23.890332 | 23.495427 | -0.39490578 |

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
