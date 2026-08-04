# DUAL-INR-DIR common regularity: case 02

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.0055609264 | 0.0054322587 | -0.00012866776 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.004162477 | 0.0041226326 | -3.9844363e-05 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.004496562 | 0.0043168748 | -0.00017968729 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0018099767 | 0.0017756267 | -3.4350041e-05 |
| `log_jdet_std_full` | dimensionless | 0.1065404 | 0.10687196 | 0.00033155631 |
| `log_jdet_std_fg` | dimensionless | 0.078669803 | 0.079208651 | 0.000538848 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.95457308 | 0.95529046 | 0.00071738073 |
| `jdet_mean_fg` | dimensionless | 0.92127982 | 0.92178541 | 0.00050558772 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 2.700011 | 2.6733648 | -0.026646194 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 6.9882439 | 6.9513452 | -0.036898757 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 10.0198 | 9.9786267 | -0.041173824 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 2.775054 | 2.7701061 | -0.004947859 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 6.0589571 | 6.0668041 | 0.0078470585 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 9.9273195 | 9.9432322 | 0.015912662 |
| `disp_mag_mean_full_mm` | mm | 5.5522188 | 5.4957767 | -0.056442066 |
| `disp_mag_p95_full_mm` | mm | 16.824515 | 16.779079 | -0.045436104 |
| `disp_mag_max_full_mm` | mm | 24.637497 | 24.512795 | -0.12470133 |
| `disp_mag_mean_fg_mm` | mm | 5.7983584 | 5.78645 | -0.011908362 |
| `disp_mag_p95_fg_mm` | mm | 14.451231 | 14.46244 | 0.011208185 |
| `disp_mag_max_fg_mm` | mm | 24.329928 | 24.326002 | -0.0039258848 |

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
