# IMPORTANT: LungCT IDIR vs pTVReg Protocol Bias

This note is intentionally placed at the top of `codex_auto_exp` because it affects
comparisons across task folders. Treat it as a protocol warning before interpreting
LungCT 4DCT/COPD TRE tables.

## Critical Warning

**Do not treat original IDIR, pTVReg, and our pTV-preprocessed SINR/DirectCP runs
as a clean method-only comparison unless the input preprocessing protocol is matched.**

The TRE values are numerically comparable in mm only because they are evaluated on
the original DIR-Lab/COPD landmarks in the original physical coordinate system. That
does not make the optimization problems equivalent. Native full-resolution IDIR and
1 mm landmark-cropped pTV-style runs see different images, different spatial domains,
different masks, different effective regularization scales, and different interpolation
histories.

Use explicit protocol tags in reports, for example:

- `native_full_lungmask_idir`
- `ptv_landmark_crop_1mm_no_lungmask`
- `native_landmark_crop_no_lungmask`
- `ptv_landmark_crop_1mm_with_idir_loss`

## Original IDIR Protocol

Original IDIR uses the native DIR-Lab 4DCT images directly.

- Dataset in paper/repo: DIR-Lab 4DCT, not COPD.
- Image files: `case{}_T00_s.img` and `case{}_T50_s.img`.
- Input shape: native per-case array shape `[z, y, x]`.
- Spacing: native per-case voxel spacing; used for TRE in mm, not for resampling.
- Resampling: **none** in the original repo path.
- Crop: **none**; the full native image tensor is used.
- Optimization domain: random samples from a lung mask on the fixed/inspiration image.
- Lung-mask convention: **fixed/T00 mask only**. The original code loads
  `Masks\case*_T00_s.mhd` and does not load a moving/T50 mask. That fixed mask is
  used to choose training coordinates in fixed-image space; the moving image is then
  sampled after transforming those coordinates.
- Coordinates: native voxel lattice is normalized to `[-1, 1]^3` for the INR.
- Similarity loss: sampled global NCC over 10,000 masked points per epoch.
- Final reported regularization: SIREN with bending energy only.
- Not used in final IDIR row: Jacobian and hyperelastic regularization, although the
  paper discusses them.
- IDIR bending energy: analytic/autograd BE on sampled normalized coordinates, not the
  same as dense voxel-grid forward-difference BE used in task9-style evaluation metrics.

Key source files:

- IDIR data loader and native shapes/spacings: `../external/IDIR/utils/general.py`
- IDIR training loop and random masked sampling: `../external/IDIR/models/models.py`
- IDIR NCC: `../external/IDIR/objectives/ncc.py`
- IDIR analytic regularizers: `../external/IDIR/objectives/regularizers.py`
- IDIR default final script: `../external/IDIR/run.py`
- Our IDIR reproduction wrapper: `task4_lungct/scripts/run_idir_lungct_case.py`

Fixed-mask evidence:

- The official loader reads only the fixed/inspiration mask:
  `../external/IDIR/utils/general.py:92-94`.
- IDIR converts that mask into possible training coordinates using the fixed image
  shape: `../external/IDIR/models/models.py:230-232`.
- During training, fixed intensities are sampled at those fixed-domain coordinates
  and compared to the transformed moving image: `../external/IDIR/models/models.py:318-327`.
- Our native IDIR wrapper keeps the same convention with
  `caseXX_T00_R231_binary_zyx.npy`: `task4_lungct/scripts/run_idir_lungct_case.py:142-175`.
- Our pTV-preprocessed IDIR wrapper applies the same crop plus nearest-neighbor
  resize to the fixed/T00 mask: `task4_lungct/scripts/run_idir_ptvprep_lungct_case.py:2-7`
  and `task4_lungct/scripts/run_idir_ptvprep_lungct_case.py:304-326`.

Visualization implication: if a visual audit overlays this same mask on the moving
`T50` image, that overlay is showing the **IDIR fixed-domain sampling mask**, not a
moving-phase lung segmentation. A separate moving mask can be useful for anatomical
quality control, but it is not part of the original IDIR optimization protocol.

## pTVReg-Style Lung Preprocessing

Official pTVReg and our pTV-style LungCT runners use a different image protocol.

- 4DCT fixed/moving convention: `T00 <- T50`.
- COPD fixed/moving convention: `iBH <- eBH`.
- Intensity: clip CT to `[80, 900]`, subtract `80`, divide by `820`.
- Crop: landmark bounding box over fixed and moving landmarks with margin
  `[x, y, z] = [10, 10, 5]`, implemented as `[z, y, x] = [5, 10, 10]`.
- Resampling: main pTVReg configuration resamples the crop to approximately
  `1 x 1 x 1 mm^3` with target shape `round(crop_shape_zyx * native_spacing_zyx)`.
- Lung mask: **not used** for pTVReg official DIR/COPD examples.
- pTVReg official mask-like behavior: `border_mask = 5`, excluding a 5-voxel border
  from the image metric.
- Our SINR/DirectCP pTV-prep runs: no lung mask for optimization; `fixed > 0` after
  pTV intensity thresholding is used only as a foreground mask for reporting regularity
  metrics, not as a similarity-loss mask.
- pTVReg official similarity: local NCC with sigma `2.1 mm`.
- Our OASIS-style SINR/DirectCP pTV-prep similarity: global NCC on the full
  pTV-preprocessed registration crop.

Key source files:

- Official pTVReg 4DCT script: `../external/pTVreg/examples_ptv/DIR_test_all.m`
- Official pTVReg COPD script: `../external/pTVreg/examples_ptv/COPD_finetune.m`
- pTVReg input processing and border mask: `../external/pTVreg/ptv/ptv_process_input.m`
- pTVReg documentation note about no lung masks: `../external/pTVreg/Examples.md`
- Our pTV-prep SINR runner: `task4_lungct/scripts/run_sinr_oasis_cps4_ptvprep_lungct_case.py`
- Our matched DirectCP runner: `task4_lungct/scripts/run_directcp_oasis_cps4_ptvprep_lungct_case.py`
- Our multistage DirectCP runner: `task4_lungct/scripts/run_multistage_directcp_oasis_ptvprep_lungct_case.py`

## Protocol-Level Input Comparison

| Property | Original IDIR 4DCT | Official pTVReg 4DCT/COPD | Our SINR/DirectCP pTV-prep |
|---|---|---|---|
| Main input image domain | Full native volume | Landmark crop | Landmark crop |
| Resampling before optimization | No | Yes for main/best reported 1 mm configs | Yes by default |
| Registration spacing | Native anisotropic spacing | Approx. `1 x 1 x 1 mm^3` after resampling | Approx. `1 x 1 x 1 mm^3` after resampling |
| Native z spacing impact | Kept as native tensor axis; z spacing only enters TRE | z axis is upsampled from 2.5 mm slices to about 1 mm | z axis is upsampled from 2.5 mm slices to about 1 mm |
| Crop around landmarks | No | Yes, `[x,y,z]=[10,10,5]` margin | Yes, `[x,y,z]=[10,10,5]` margin |
| Lung mask for optimization | Yes, fixed/T00 sampled lung-mask domain only; no moving/T50 mask | No | No |
| Border exclusion | No explicit pTV-style border mask | `border_mask = 5` | pTV torch uses border mask; OASIS SINR/DirectCP uses full crop |
| Similarity | Sampled global NCC | Local NCC, sigma `2.1 mm` | Global NCC for OASIS SINR/DirectCP pTV-prep |
| Final reported regularizer | Analytic sampled-coordinate BE | isoTV on control points | Dense displacement L2 for OASIS SINR/DirectCP pTV-prep |
| TRE evaluation | Original landmarks, native physical spacing | Original landmarks, native physical spacing after resizing/uncrop | Original landmarks, native physical spacing after resizing/uncrop |

## pTVReg Original vs 1 mm Resampled 4DCT Results

The pTVReg repo explicitly evaluated DIR-Lab 4DCT at both original resolution and
1 mm resampled resolution. The script `../external/pTVreg/examples_ptv/DIR_test_all.m`
sweeps `resize = 0:1`, but skips `cp_refinements = 1` when `resize = 0`. Therefore
the clean original-vs-resampled comparison is only for `cp_refinements = 0`.

Values below are read from the official saved artifact
`../external/pTVreg/examples_ptv/results/DIR_TRE.mat`. The TRE values match the repo
Markdown tables up to rounding; the mean times below are computed from the saved
artifact and are close to, but not always identical to, the rounded Markdown times.

| pTVReg 4DCT config | Resolution | Mean TRE mm | Mean time sec | Per-case TRE mm |
|---|---|---:|---:|---|
| `cp_refinements=0`, `loc_cc_approximate=false` | Original | `0.962` | `74.9` | `0.824, 0.830, 0.977, 1.284, 1.126, 0.883, 0.827, 1.048, 0.946, 0.872` |
| `cp_refinements=0`, `loc_cc_approximate=false` | 1 mm resampled | `0.930` | `191.5` | `0.804, 0.768, 0.921, 1.283, 1.110, 0.807, 0.796, 1.120, 0.905, 0.786` |
| `cp_refinements=0`, `loc_cc_approximate=true` | Original | `0.958` | `60.0` | `0.780, 0.791, 0.956, 1.269, 1.094, 0.888, 0.861, 1.069, 0.944, 0.929` |
| `cp_refinements=0`, `loc_cc_approximate=true` | 1 mm resampled | `0.918` | `133.2` | `0.767, 0.747, 0.927, 1.259, 1.072, 0.834, 0.800, 1.014, 0.911, 0.844` |
| `cp_refinements=1`, `loc_cc_approximate=false` | 1 mm resampled only | `0.922` | `304.3` | `0.801, 0.768, 0.924, 1.296, 1.127, 0.784, 0.793, 0.996, 0.909, 0.820` |
| `cp_refinements=1`, `loc_cc_approximate=true` | 1 mm resampled only | `0.920` | `180.5` | `0.776, 0.743, 0.919, 1.272, 1.087, 0.837, 0.807, 0.990, 0.918, 0.848` |

For the like-for-like `cp_refinements=0` comparison, 1 mm resampling improves pTVReg
mean TRE by about `0.032 mm` with exact LCC gradients and `0.040 mm` with approximate
LCC gradients. Runtime increases substantially: about `+116.6 sec` for exact LCC and
`+73.2 sec` for approximate LCC.

Important nuance: pTVReg's original-resolution rows are still landmark-cropped,
thresholded, no-lung-mask pTVReg runs. They are **not** equivalent to original IDIR,
which uses the full native image and lung-mask sampled optimization.

## pTVReg Image Pyramid vs Control-Point Refinement

pTVReg has two separate multiresolution mechanisms.

1. Image pyramid stages: many levels are built with `k_down = 0.7`.
2. Control-point refinement stages: the lung examples use at most one extra refinement,
   e.g. `8 x 8 x 8 -> 4 x 4 x 4`.

For the 4DCT headline setting (`cp_refinements=0`, 1 mm resampled), pTVReg keeps the
nominal control grid spacing at `4 x 4 x 4` pixels throughout the image pyramid. This
does **not** mean the physical control spacing is constant across levels: the image is
downsampled, and pTVReg updates `pix_resolution(i) = pix_resolution(i-1) / k_down`.
Thus `4` pixels at a coarser pyramid level corresponds to a larger physical spacing, and
the number of control points is smaller because the image array is smaller.

For COPD best/fine-tuned and the optional refined 4DCT rows, pTVReg first optimizes
with nominal `8 x 8 x 8` control spacing through the image pyramid, then halves the
spacing once at the finest image level to `4 x 4 x 4`.

Mechanically, `ptv_register.m` loops over `nlvl + cp_refinements`. During the image
pyramid part it sets `cur_grid_spacing = grid_spacing`; during refinement it sets
`cur_grid_spacing = round(cur_grid_spacing / 2)`. Knots are carried between stages
with variational upsampling.

The image pyramid itself is not just blurring. In the default lung path
(`fine_pyramid=false`):

- each next level is Gaussian-smoothed from the previous level;
- for `k_down=0.7`, the smoothing sigma is
  `0.4 * (0.5 / 0.7) ~= 0.286` pixels per axis;
- the smoothed image is then actually resized/downsampled to
  `ceil(0.7 * previous_shape)`;
- voxel spacing is increased by `1 / 0.7` per level;
- the lung examples pass `interp_type=0`, which selects the linear interpolation path
  for resizing/warping helpers.

This means pTVReg's good 4DCT result with nominal `4 x 4 x 4` spacing is still a
many-stage coarse-to-fine image-pyramid optimization, not a single-scale 4 mm
optimization.

## 4DCT Input Size Comparison

Shapes below are `[z, y, x]`. The pTV crop shapes are computed from the real landmark
files under `/data/hengjie/datasets/lungCT` using the same crop formula as our runners.
The pTV 1 mm registration shape is `round(crop_shape_zyx * native_spacing_zyx)`.

| Case | IDIR/native full shape z,y,x | Native spacing z,y,x mm | pTV landmark crop shape z,y,x | pTV 1mm reg shape z,y,x | Crop volume / native volume |
|---:|---:|---:|---:|---:|---:|
| 1 | `[94, 256, 256]` | `[2.5, 0.97, 0.97]` | `[84, 155, 239]` | `[210, 150, 232]` | `0.505` |
| 2 | `[112, 256, 256]` | `[2.5, 1.16, 1.16]` | `[101, 168, 241]` | `[252, 195, 280]` | `0.557` |
| 3 | `[104, 256, 256]` | `[2.5, 1.15, 1.15]` | `[96, 151, 236]` | `[240, 174, 271]` | `0.502` |
| 4 | `[99, 256, 256]` | `[2.5, 1.13, 1.13]` | `[91, 159, 216]` | `[228, 180, 244]` | `0.482` |
| 5 | `[106, 256, 256]` | `[2.5, 1.1, 1.1]` | `[91, 165, 231]` | `[228, 182, 254]` | `0.499` |
| 6 | `[128, 512, 512]` | `[2.5, 0.97, 0.97]` | `[92, 166, 270]` | `[230, 161, 262]` | `0.123` |
| 7 | `[136, 512, 512]` | `[2.5, 0.97, 0.97]` | `[87, 184, 284]` | `[218, 178, 275]` | `0.128` |
| 8 | `[128, 512, 512]` | `[2.5, 0.97, 0.97]` | `[91, 213, 262]` | `[228, 207, 254]` | `0.151` |
| 9 | `[128, 512, 512]` | `[2.5, 0.97, 0.97]` | `[66, 186, 233]` | `[165, 180, 226]` | `0.085` |
| 10 | `[120, 512, 512]` | `[2.5, 0.97, 0.97]` | `[84, 215, 216]` | `[210, 209, 210]` | `0.124` |

The crop-volume ratio is especially important: for 4DCT cases 6-10, pTV-style
optimization uses only about `8.5%` to `15.1%` of the native full volume. This is not
the same optimization domain as IDIR's full native image with lung-mask sampling.

## COPD pTV-Style Input Sizes

Original IDIR did not report COPD in the inspected paper/repo path. The COPD table is
included because our lung CT tasks also run pTV-style SINR/DirectCP on COPD. Shapes
below are again `[z, y, x]`.

| Case | Native full shape z,y,x | Native spacing z,y,x mm | pTV landmark crop shape z,y,x | pTV 1mm reg shape z,y,x | Crop volume / native volume |
|---:|---:|---:|---:|---:|---:|
| 1 | `[121, 512, 512]` | `[2.5, 0.625, 0.625]` | `[110, 321, 366]` | `[275, 201, 229]` | `0.407` |
| 2 | `[102, 512, 512]` | `[2.5, 0.645, 0.645]` | `[86, 306, 382]` | `[215, 197, 246]` | `0.376` |
| 3 | `[126, 512, 512]` | `[2.5, 0.652, 0.652]` | `[101, 289, 405]` | `[252, 188, 264]` | `0.358` |
| 4 | `[126, 512, 512]` | `[2.5, 0.59, 0.59]` | `[101, 293, 395]` | `[252, 173, 233]` | `0.354` |
| 5 | `[131, 512, 512]` | `[2.5, 0.647, 0.647]` | `[105, 298, 383]` | `[262, 193, 248]` | `0.349` |
| 6 | `[119, 512, 512]` | `[2.5, 0.633, 0.633]` | `[101, 284, 388]` | `[252, 180, 246]` | `0.357` |
| 7 | `[112, 512, 512]` | `[2.5, 0.625, 0.625]` | `[102, 267, 402]` | `[255, 167, 251]` | `0.373` |
| 8 | `[115, 512, 512]` | `[2.5, 0.586, 0.586]` | `[100, 299, 409]` | `[250, 175, 240]` | `0.406` |
| 9 | `[116, 512, 512]` | `[2.5, 0.664, 0.664]` | `[96, 287, 398]` | `[240, 191, 264]` | `0.361` |
| 10 | `[135, 512, 512]` | `[2.5, 0.742, 0.742]` | `[93, 286, 401]` | `[232, 212, 298]` | `0.301` |

## Why This Can Bias TRE

Important confounders:

- Resampling to 1 mm changes the image signal, especially through-plane where native
  spacing is usually `2.5 mm`.
- Landmark cropping focuses optimization around the TRE evaluation region.
- IDIR's lung mask and pTVReg's no-lung-mask crop are different spatial sampling domains.
- Control spacing has different physical meaning across native and 1 mm inputs. For
  example, `cps=4` after 1 mm resampling is about `4 mm`; `cps=4` on native 4DCT would
  be about `10 mm` through-plane.
- Regularizer magnitudes are not directly comparable across native anisotropic grids,
  normalized INR coordinates, and 1 mm isotropic crops.
- Dense-field evaluation metrics such as task9 forward BE/diffusion/Jdet are valid
  posthoc deformation metrics, but they are not necessarily the regularizers optimized by
  IDIR or pTVReg.

## Recommended Fair Comparisons

For any LungCT method comparison, separate benchmark comparison from method-only
ablation.

Benchmark-style comparison is acceptable if clearly labeled:

- IDIR reproduced: `native_full_lungmask_idir`
- pTVReg official: `ptv_landmark_crop_1mm_no_lungmask`
- SINR/DirectCP pTV-prep: `ptv_landmark_crop_1mm_no_lungmask`

Method-only ablation should include crossed protocols:

- IDIR original: native full image, lung-mask sampled NCC, analytic BE.
- IDIR on pTV-preprocessed 1 mm crop.
- SINR/DirectCP on pTV-preprocessed 1 mm crop.
- SINR/DirectCP on an IDIR-style native/lung-mask protocol, if feasible.

Until those crossed runs are done, do not attribute an observed TRE gap solely to the
deformation parameterization or optimizer.
