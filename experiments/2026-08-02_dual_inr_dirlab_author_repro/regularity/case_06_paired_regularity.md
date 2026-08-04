# DUAL-INR-DIR common regularity: case 06

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.018544899 | 0.018302272 | -0.00024262689 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.020530717 | 0.020337739 | -0.0001929781 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.0085027271 | 0.0084470761 | -5.5650981e-05 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0062149622 | 0.0061880765 | -2.6885697e-05 |
| `log_jdet_std_full` | dimensionless | 0.20754521 | 0.20685052 | -0.00069469236 |
| `log_jdet_std_fg` | dimensionless | 0.16787834 | 0.16702852 | -0.0008498175 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.92033589 | 0.91989644 | -0.00043945006 |
| `jdet_mean_fg` | dimensionless | 0.78989014 | 0.79150702 | 0.0016168775 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 6.2911429 | 6.2951674 | 0.0040245108 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 13.277279 | 13.15795 | -0.11932898 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 18.991367 | 18.789905 | -0.20146287 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 6.4782647 | 6.4726855 | -0.0055792196 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 13.241652 | 13.211946 | -0.029705967 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 18.860429 | 18.462752 | -0.39767772 |
| `disp_mag_mean_full_mm` | mm | 10.81082 | 10.765373 | -0.045446867 |
| `disp_mag_p95_full_mm` | mm | 24.406638 | 24.282598 | -0.1240403 |
| `disp_mag_max_full_mm` | mm | 30.280847 | 30.350057 | 0.069209151 |
| `disp_mag_mean_fg_mm` | mm | 11.985945 | 11.975245 | -0.010700749 |
| `disp_mag_p95_fg_mm` | mm | 24.342908 | 24.326935 | -0.015973199 |
| `disp_mag_max_fg_mm` | mm | 29.099169 | 29.173275 | 0.074105496 |

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
