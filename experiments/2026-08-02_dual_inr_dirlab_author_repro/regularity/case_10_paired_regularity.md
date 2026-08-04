# DUAL-INR-DIR common regularity: case 10

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.021644881 | 0.021024987 | -0.00061989389 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.023194486 | 0.022751274 | -0.00044321272 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.01165043 | 0.011031503 | -0.0006189274 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0064640876 | 0.0062798418 | -0.00018424578 |
| `log_jdet_std_full` | dimensionless | 0.18760658 | 0.18902351 | 0.0014169365 |
| `log_jdet_std_fg` | dimensionless | 0.15885476 | 0.15921851 | 0.00036375302 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.95012121 | 0.95322073 | 0.0030995262 |
| `jdet_mean_fg` | dimensionless | 0.86453554 | 0.86563913 | 0.0011035991 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 4.648533 | 4.6503632 | 0.0018301817 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 11.519857 | 11.474918 | -0.044939348 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 18.358783 | 18.012431 | -0.34635212 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 4.8606496 | 4.8411307 | -0.01951886 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 11.675925 | 11.674296 | -0.00162857 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 18.338617 | 17.999647 | -0.33896934 |
| `disp_mag_mean_full_mm` | mm | 8.2490571 | 8.2615943 | 0.012537272 |
| `disp_mag_p95_full_mm` | mm | 20.645295 | 20.629691 | -0.015604043 |
| `disp_mag_max_full_mm` | mm | 29.594644 | 29.111392 | -0.48325217 |
| `disp_mag_mean_fg_mm` | mm | 9.4410106 | 9.394708 | -0.046302623 |
| `disp_mag_p95_fg_mm` | mm | 22.211553 | 22.23995 | 0.028396911 |
| `disp_mag_max_fg_mm` | mm | 28.20536 | 28.657968 | 0.45260829 |

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
