# DUAL-INR-DIR common regularity: case 09

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.013754332 | 0.013620817 | -0.00013351528 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.013891783 | 0.013697751 | -0.00019403165 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.012916814 | 0.012802859 | -0.00011395542 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0078141321 | 0.007708176 | -0.00010595602 |
| `log_jdet_std_full` | dimensionless | 0.16210341 | 0.1620822 | -2.1211005e-05 |
| `log_jdet_std_fg` | dimensionless | 0.16258434 | 0.16125811 | -0.0013262299 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.93735278 | 0.94004918 | 0.002696402 |
| `jdet_mean_fg` | dimensionless | 0.86791275 | 0.86936138 | 0.0014486328 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 3.8619224 | 3.8731664 | 0.011243979 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 7.6948314 | 7.7075864 | 0.012755088 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 12.060922 | 11.4525 | -0.60842266 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 4.4115025 | 4.4014866 | -0.010015876 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 7.8216402 | 7.8085884 | -0.013051843 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 11.303995 | 10.845007 | -0.45898844 |
| `disp_mag_mean_full_mm` | mm | 6.6291097 | 6.628332 | -0.00077775103 |
| `disp_mag_p95_full_mm` | mm | 13.096881 | 13.15642 | 0.05953927 |
| `disp_mag_max_full_mm` | mm | 16.99496 | 17.524824 | 0.52986434 |
| `disp_mag_mean_fg_mm` | mm | 7.3307676 | 7.3100613 | -0.020706244 |
| `disp_mag_p95_fg_mm` | mm | 13.581999 | 13.582175 | 0.00017590308 |
| `disp_mag_max_fg_mm` | mm | 16.683774 | 16.93625 | 0.25247576 |

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
