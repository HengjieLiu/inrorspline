# LungCT pTV-Prep No-Sweep Reproduction Report

Generated from completed `summary.json` files under:

`/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1`

## Mean Results

| Method | Cases | Original Snap TRE | Original Sub TRE | 1mm Snap TRE | 1mm Sub TRE | FG logJ std | FG J<=0% | BE fg | Diff fg | Time sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| idir_be_default_lungmask | 10 | 1.017 +/- 0.176 | 1.133 +/- 0.137 | 1.236 +/- 0.133 | 1.138 +/- 0.136 | 0.142869 | 0.0000 | 0.000535 | 0.007218 | 157.6 |
| sinr_cps4_ptv_lcc_isotv | 10 | 2.904 +/- 3.209 | 3.014 +/- 3.178 | 3.143 +/- 3.201 | 3.037 +/- 3.206 | 0.301655 | 0.8918 | 0.006339 | 0.023879 | 154.2 |
| direct_cp_cps4_ptv_lcc_isotv | 10 | 6.370 +/- 3.340 | 6.450 +/- 3.302 | 6.575 +/- 3.316 | 6.495 +/- 3.329 | 0.652560 | 3.9433 | 0.034067 | 0.069535 | 112.6 |
| direct_cp_32_16_8_4_ptv_lcc_isotv | 10 | 0.922 +/- 0.152 | 1.049 +/- 0.119 | 1.168 +/- 0.114 | 1.052 +/- 0.118 | 0.179961 | 0.0138 | 0.002527 | 0.010327 | 234.1 |

Original TRE means native/crop-space TRE: the saved 1mm displacement field is resized back to the corrected pTVReg native landmark crop, converted to native voxel units, and evaluated with native spacing. This is the pTVReg-comparable headline convention. 1mm TRE is evaluated directly in the cropped 1mm registration grid.

## Legacy Reference Rows

Reference rows are comparison anchors only. Their TRE columns are old task4 native/crop snap/sub values. The new 4-stage run uses stage epochs `200,400,800,1000`, so it is not byte-identical to the legacy full or reduced schedules, and the canonical regenerated crop uses the corrected pTVReg crop-origin convention.

| Reference | Original Snap TRE | Original Sub TRE | FG logJ std | FG J<=0% | BE fg | Diff fg | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IDIR iso-1mm + lung-mask sampled NCC | 1.132 | 1.212 | 0.139823 | 0.0000 | 0.004565 | 0.013311 | legacy task4 pTV-prep IDIR row |
| DirectCP 32-16-8-4 + pTV LCC + isoTV full | 1.040 | 1.134 | 0.147097 | 0.0057 | 0.011850 | 0.012041 | legacy task4 300,500,800,1200 schedule |
| DirectCP 32-16-8-4 + pTV LCC + isoTV reduced lr1e-3 | 1.045 | 1.139 | 0.146800 | 0.0079 | 0.011900 | 0.012100 | legacy task4 200,200,200,400 schedule |
| DirectCP 32-16-8-4 + pTV LCC + isoTV reduced lr1e-2 | 1.035 | 1.135 | 0.149800 | 0.0075 | 0.012700 | 0.012500 | legacy task4 200,200,200,400 schedule |

## Per-Case Rows

| Case | Method | Original Identity TRE | Original Snap TRE | Original Sub TRE | 1mm Identity TRE | 1mm Snap TRE | 1mm Sub TRE | FG logJ std | FG J<=0% | BE fg | Diff fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | direct_cp_32_16_8_4_ptv_lcc_isotv | 3.892 | 0.774 | 0.933 | 3.916 | 1.068 | 0.937 | 0.100624 | 0.0000 | 0.000731 | 0.003049 |
| 2 | direct_cp_32_16_8_4_ptv_lcc_isotv | 4.338 | 0.775 | 0.928 | 4.354 | 1.068 | 0.930 | 0.104508 | 0.0000 | 0.001245 | 0.004700 |
| 3 | direct_cp_32_16_8_4_ptv_lcc_isotv | 6.943 | 0.923 | 1.063 | 6.981 | 1.161 | 1.068 | 0.119684 | 0.0001 | 0.001223 | 0.005361 |
| 4 | direct_cp_32_16_8_4_ptv_lcc_isotv | 9.830 | 1.275 | 1.320 | 9.912 | 1.435 | 1.316 | 0.210633 | 0.0000 | 0.002706 | 0.010194 |
| 5 | direct_cp_32_16_8_4_ptv_lcc_isotv | 7.477 | 1.087 | 1.202 | 7.536 | 1.305 | 1.206 | 0.227205 | 0.0394 | 0.002974 | 0.011053 |
| 6 | direct_cp_32_16_8_4_ptv_lcc_isotv | 10.891 | 0.836 | 1.018 | 10.953 | 1.139 | 1.021 | 0.235830 | 0.0255 | 0.003905 | 0.014033 |
| 7 | direct_cp_32_16_8_4_ptv_lcc_isotv | 11.026 | 0.806 | 0.969 | 11.114 | 1.107 | 0.975 | 0.191859 | 0.0012 | 0.002753 | 0.011807 |
| 8 | direct_cp_32_16_8_4_ptv_lcc_isotv | 14.995 | 1.005 | 1.085 | 15.109 | 1.214 | 1.082 | 0.217253 | 0.0566 | 0.003529 | 0.017057 |
| 9 | direct_cp_32_16_8_4_ptv_lcc_isotv | 7.918 | 0.910 | 1.011 | 7.968 | 1.123 | 1.018 | 0.178573 | 0.0010 | 0.002687 | 0.010902 |
| 10 | direct_cp_32_16_8_4_ptv_lcc_isotv | 7.301 | 0.827 | 0.964 | 7.349 | 1.062 | 0.964 | 0.213441 | 0.0144 | 0.003519 | 0.015115 |
| 1 | direct_cp_cps4_ptv_lcc_isotv | 3.892 | 1.176 | 1.331 | 3.916 | 1.466 | 1.339 | 0.268250 | 0.7862 | 0.007803 | 0.015028 |
| 2 | direct_cp_cps4_ptv_lcc_isotv | 4.338 | 2.981 | 3.098 | 4.354 | 3.258 | 3.109 | 0.556620 | 2.3723 | 0.022467 | 0.045705 |
| 3 | direct_cp_cps4_ptv_lcc_isotv | 6.943 | 4.186 | 4.305 | 6.981 | 4.361 | 4.330 | 0.571654 | 2.8903 | 0.026302 | 0.055463 |
| 4 | direct_cp_cps4_ptv_lcc_isotv | 9.830 | 6.640 | 6.685 | 9.912 | 6.808 | 6.740 | 0.736270 | 5.0581 | 0.044456 | 0.092629 |
| 5 | direct_cp_cps4_ptv_lcc_isotv | 7.477 | 5.814 | 5.909 | 7.536 | 6.027 | 5.955 | 0.622089 | 3.3552 | 0.029831 | 0.061157 |
| 6 | direct_cp_cps4_ptv_lcc_isotv | 10.891 | 8.497 | 8.552 | 10.953 | 8.678 | 8.606 | 0.886695 | 7.3552 | 0.049440 | 0.102936 |
| 7 | direct_cp_cps4_ptv_lcc_isotv | 11.026 | 9.037 | 9.095 | 11.114 | 9.265 | 9.170 | 0.759342 | 5.0756 | 0.042609 | 0.085307 |
| 8 | direct_cp_cps4_ptv_lcc_isotv | 14.995 | 13.815 | 13.819 | 15.109 | 13.988 | 13.924 | 0.801341 | 5.2323 | 0.050272 | 0.098600 |
| 9 | direct_cp_cps4_ptv_lcc_isotv | 7.918 | 6.097 | 6.153 | 7.968 | 6.250 | 6.194 | 0.617874 | 3.0024 | 0.033656 | 0.066410 |
| 10 | direct_cp_cps4_ptv_lcc_isotv | 7.301 | 5.459 | 5.551 | 7.349 | 5.647 | 5.586 | 0.705469 | 4.3048 | 0.033836 | 0.072110 |
| 1 | idir_be_default_lungmask | 3.892 | 0.782 | 0.940 | 3.916 | 1.063 | 0.943 | 0.097494 | 0.0000 | 0.000339 | 0.002921 |
| 2 | idir_be_default_lungmask | 4.338 | 0.758 | 0.934 | 4.354 | 1.054 | 0.935 | 0.080090 | 0.0000 | 0.000238 | 0.003582 |
| 3 | idir_be_default_lungmask | 6.943 | 0.944 | 1.097 | 6.981 | 1.190 | 1.103 | 0.098738 | 0.0000 | 0.000277 | 0.004096 |
| 4 | idir_be_default_lungmask | 9.830 | 1.371 | 1.376 | 9.912 | 1.473 | 1.368 | 0.179309 | 0.0000 | 0.000542 | 0.007043 |
| 5 | idir_be_default_lungmask | 7.477 | 1.224 | 1.325 | 7.536 | 1.437 | 1.333 | 0.145756 | 0.0000 | 0.000586 | 0.006886 |
| 6 | idir_be_default_lungmask | 10.891 | 1.000 | 1.150 | 10.953 | 1.253 | 1.158 | 0.174312 | 0.0000 | 0.000747 | 0.009289 |
| 7 | idir_be_default_lungmask | 11.026 | 0.960 | 1.105 | 11.114 | 1.199 | 1.105 | 0.154399 | 0.0000 | 0.000602 | 0.008654 |
| 8 | idir_be_default_lungmask | 14.995 | 1.119 | 1.224 | 15.109 | 1.331 | 1.234 | 0.172968 | 0.0002 | 0.000572 | 0.012068 |
| 9 | idir_be_default_lungmask | 7.918 | 0.987 | 1.096 | 7.968 | 1.183 | 1.103 | 0.163406 | 0.0000 | 0.000794 | 0.007170 |
| 10 | idir_be_default_lungmask | 7.301 | 1.028 | 1.087 | 7.349 | 1.182 | 1.093 | 0.162219 | 0.0000 | 0.000659 | 0.010471 |
| 1 | sinr_cps4_ptv_lcc_isotv | 3.892 | 0.794 | 0.937 | 3.916 | 1.082 | 0.941 | 0.095489 | 0.0000 | 0.000501 | 0.002734 |
| 2 | sinr_cps4_ptv_lcc_isotv | 4.338 | 0.745 | 0.922 | 4.354 | 1.060 | 0.924 | 0.090729 | 0.0000 | 0.000670 | 0.003998 |
| 3 | sinr_cps4_ptv_lcc_isotv | 6.943 | 0.943 | 1.072 | 6.981 | 1.169 | 1.081 | 0.107656 | 0.0000 | 0.000704 | 0.004675 |
| 4 | sinr_cps4_ptv_lcc_isotv | 9.830 | 1.275 | 1.320 | 9.912 | 1.434 | 1.314 | 0.200036 | 0.0000 | 0.001829 | 0.009243 |
| 5 | sinr_cps4_ptv_lcc_isotv | 7.477 | 2.576 | 2.698 | 7.536 | 2.805 | 2.723 | 0.338842 | 0.8346 | 0.006291 | 0.022912 |
| 6 | sinr_cps4_ptv_lcc_isotv | 10.891 | 3.393 | 3.533 | 10.953 | 3.676 | 3.561 | 0.518423 | 2.9176 | 0.012201 | 0.043833 |
| 7 | sinr_cps4_ptv_lcc_isotv | 11.026 | 3.969 | 4.086 | 11.114 | 4.222 | 4.130 | 0.423293 | 0.9163 | 0.009324 | 0.035535 |
| 8 | sinr_cps4_ptv_lcc_isotv | 14.995 | 11.950 | 11.959 | 15.109 | 12.154 | 12.057 | 0.641020 | 2.8898 | 0.019735 | 0.070124 |
| 9 | sinr_cps4_ptv_lcc_isotv | 7.918 | 0.942 | 1.027 | 7.968 | 1.124 | 1.033 | 0.173762 | 0.0392 | 0.002129 | 0.009883 |
| 10 | sinr_cps4_ptv_lcc_isotv | 7.301 | 2.448 | 2.584 | 7.349 | 2.706 | 2.604 | 0.427298 | 1.3211 | 0.010001 | 0.035853 |

## Protocol Notes

- Inputs are canonical regenerated pTVReg-style landmark crops resampled to 1mm.
- pTVReg's own DIR examples report snap-to-voxel TRE after resizing/carrying the displacement back to the original crop/full native domain; see `external/pTVreg/examples_ptv/DIR_test_all.m`, `external/pTVreg/mutils/My/DIR_helpers/DIR_movepoints.m`, and `external/pTVreg/DIRandCOPD.md`.
- In the 1mm registration grid, landmarks are fractional after crop/resampling; snap TRE can therefore be larger than subvoxel TRE. In original/native-crop space, integer landmark targets usually make snap TRE smaller than subvoxel TRE.
- IDIR uses the fixed T00 lung mask for sampled-coordinate training.
- SINR/DirectCP pTV methods use border-masked pTV LCC plus isoTV, not lung-mask-weighted loss.
- All foreground regularity metrics use the fixed T00 lung mask on the cropped 1mm grid.
