# DUAL-INR-DIR common regularity: case 03

Seed: `42`. Paired delta is `A1 - A0`.

| Metric | Unit/convention | A0 | A1 | A1 - A0 |
|---|---|---:|---:|---:|
| `diffusion_l2_full` | voxel-coordinate finite difference | 0.0064096175 | 0.0063323014 | -7.7316087e-05 |
| `diffusion_l2_fg` | voxel-coordinate finite difference | 0.0054730416 | 0.0053357801 | -0.00013726156 |
| `bending_energy_full` | voxel-coordinate finite difference | 0.0043820349 | 0.0043755714 | -6.4634592e-06 |
| `bending_energy_fg` | voxel-coordinate finite difference | 0.0019806135 | 0.0019333285 | -4.7285015e-05 |
| `log_jdet_std_full` | dimensionless | 0.1148466 | 0.11365516 | -0.0011914434 |
| `log_jdet_std_fg` | dimensionless | 0.099210603 | 0.098660961 | -0.00054964172 |
| `jdet_le0_pct_full` | % | 0 | 0 | 0 |
| `jdet_le0_pct_fg` | % | 0 | 0 | 0 |
| `jdet_mean_full` | dimensionless | 0.94950021 | 0.95003997 | 0.00053975834 |
| `jdet_mean_fg` | dimensionless | 0.90090516 | 0.90174576 | 0.00084059794 |
| `disp_mag_mean_full_vox` | voxel-coordinate magnitude | 3.1384033 | 3.1214685 | -0.016934809 |
| `disp_mag_p95_full_vox` | voxel-coordinate magnitude | 6.3088889 | 6.3083105 | -0.00057844032 |
| `disp_mag_max_full_vox` | voxel-coordinate magnitude | 8.4796475 | 8.4043586 | -0.075288897 |
| `disp_mag_mean_fg_vox` | voxel-coordinate magnitude | 3.4773014 | 3.4765753 | -0.00072616512 |
| `disp_mag_p95_fg_vox` | voxel-coordinate magnitude | 6.3163591 | 6.3266407 | 0.010281575 |
| `disp_mag_max_fg_vox` | voxel-coordinate magnitude | 8.2865246 | 8.2990793 | 0.012554714 |
| `disp_mag_mean_full_mm` | mm | 6.1485522 | 6.0994467 | -0.049105567 |
| `disp_mag_p95_full_mm` | mm | 14.527934 | 14.465907 | -0.062027774 |
| `disp_mag_max_full_mm` | mm | 18.988473 | 19.155806 | 0.1673332 |
| `disp_mag_mean_fg_mm` | mm | 7.1312012 | 7.1264595 | -0.0047417326 |
| `disp_mag_p95_fg_mm` | mm | 14.549915 | 14.557369 | 0.0074537912 |
| `disp_mag_max_fg_mm` | mm | 18.910174 | 18.961952 | 0.051778653 |

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
