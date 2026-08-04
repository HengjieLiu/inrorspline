# DUAL-INR-DIR common regularity: case 07

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.01459696 | 0.014379777 | -0.00021718318 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.013373877 | 0.013270065 | -0.00010381194 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.0084507375 | 0.0084740323 | 2.3294818e-05 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0047897782 | 0.0047185807 | -7.1197487e-05 |
| `log_jdet_std_full` | dimensionless | 0.18779838 | 0.18340674 | -0.0043916404 |
| `log_jdet_std_fg` | dimensionless | 0.15323913 | 0.15224612 | -0.000993016 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.90726825 | 0.90558053 | -0.0016877248 |
| `jdet_mean_fg` | dimensionless | 0.82804869 | 0.82902492 | 0.00097623594 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 5.7402033 | 5.7386949 | -0.0015084195 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 11.882345 | 11.855388 | -0.026957658 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 16.302879 | 15.945022 | -0.35785692 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 5.715054 | 5.6961861 | -0.018867954 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 11.564741 | 11.567394 | 0.002653195 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 16.264441 | 15.935276 | -0.32916512 |
| `disp_mag_mean_full_mm` | mm | 11.855742 | 11.844574 | -0.011167486 |
| `disp_mag_p95_full_mm` | mm | 27.471229 | 27.362446 | -0.10878269 |
| `disp_mag_max_full_mm` | mm | 35.015404 | 35.128944 | 0.1135398 |
| `disp_mag_mean_fg_mm` | mm | 11.897567 | 11.853833 | -0.043734358 |
| `disp_mag_p95_fg_mm` | mm | 26.631435 | 26.625625 | -0.0058098769 |
| `disp_mag_max_fg_mm` | mm | 34.971753 | 34.949711 | -0.022042539 |

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
