# IMPORTANT: pTVReg Technique Audit and DirectCP Multistage Comparison

Date: 2026-06-24

This report is intentionally placed directly under `codex_auto_exp` because the pTVReg
protocol and technique details affect multiple LungCT task folders. It complements
`IMPORTANT_LUNGCT_IDIR_PTVREG_PROTOCOL_BIAS.md`.

All pTVReg claims below were re-checked against the local official pTVReg checkout at
`external/pTVreg`. Source references use `path:line-line`.

## Executive Takeaways

1. pTVReg's strong LungCT performance is not just "direct CP cps4 optimization."
   It is a specific bundle: pTV-style CT preprocessing, landmark crop, usually 1 mm
   crop resampling, many image-pyramid levels, local CC, isoTV control regularization,
   minFunc/L-BFGS optimization, and linear displacement control grids.
2. For official 4DCT, the headline result uses nominal `4 x 4 x 4` control spacing
   with `cp_refinements=0`, but it is still a many-level image-pyramid optimization
   (`k_down=0.7`). It is not a single-scale cps4 run.
3. For official COPD best/fine-tuned, pTVReg uses `8 x 8 x 8 -> 4 x 4 x 4`
   control-grid refinement at the finest image level.
4. pTVReg does not use analytical knot insertion for the lung scripts. With
   `spline_order=1`, it transfers between levels by resizing the dense displacement
   field and variationally fitting a new linear control grid.
5. Our exact-knot DirectCP multistage matches the broad idea of coarse-to-fine
   DirectCP and roughly matches pTV-style lung preprocessing, but it does not match
   pTVReg's local CC, isoTV, linear basis, L-BFGS, pyramid schedule, border-mask
   behavior, or inter-level transfer.
6. A follow-up all-10 4DCT ablation found that adding a 5-voxel border mask to the
   `16-8-4` global-NCC DirectCP schedule modestly improves mean TRE (`1.236 -> 1.176
   mm`). The first pTV-like LCC+isoTV DirectCP variant is not yet competitive because
   case 8 fails badly (`7.391 mm`), so matching pTVReg's loss terms alone is
   insufficient.

## Official pTVReg Lung Protocol

### 4DCT DIR-Lab

The official 4DCT script sweeps `use_refinement`, `resize`, and `fast_lcc`
(`external/pTVreg/examples_ptv/DIR_test_all.m:9-11`) and skips refinement at
native/original resolution (`DIR_test_all.m:12-13`).

Main input operations:

| Item | pTVReg 4DCT behavior | Source |
|---|---|---|
| Fixed/moving phases | fixed `T00`, moving `T50` | `DIR_test_all.m:17-23` |
| Intensity | clamp to `[80,900]`, subtract 80, divide by 820 | `DIR_test_all.m:25-26`, `external/pTVreg/mutils/My/image_utils/img_thr.m:1-7` |
| Crop | landmark bounding box over fixed and moving landmarks, margin `[x,y,z]=[10,10,5]` | `DIR_test_all.m:28-38` |
| 1 mm resampling | if `resize`, target shape `round(crop_size .* spacing)`, then set spacing to `[1,1,1]` | `DIR_test_all.m:40-47` |
| Main/headline grid | if resized and no refinement, `grid_spacing=[4,4,4]`, `cp_refinements=0` | `DIR_test_all.m:59-65` |
| Optional refined grid | if resized and refined, start `grid_spacing=[8,8,8]`, `cp_refinements=1` | `DIR_test_all.m:52-58` |
| Native/original grid | if not resized, `grid_spacing=[4,4,3]`, `cp_refinements=0` | `DIR_test_all.m:59-65` |
| Metric/regularizer | `loc_cc_fftn_gpu`, sigma 2.1 mm, `isoTV=0.11`, `csqrt=5e-3` | `DIR_test_all.m:67-79` |
| TRE evaluation | resize displacement back to crop/native, uncrop, evaluate original landmarks | `DIR_test_all.m:85-93` |

The pTVReg documentation says the 4DCT headline configuration is 1 mm resampling,
`4 x 4 x 4` control spacing, LCC sigma 2.1 mm, and isoTV 0.11
(`external/pTVreg/DIRandCOPD.md:14-18`). The headline table reports mean TRE
`0.92 mm` and mean time `130 sec` (`DIRandCOPD.md:3-6`).
pTVReg's TRE helper rounds moved landmark positions before computing physical TRE
(`external/pTVreg/mutils/My/DIR_helpers/DIR_movepoints.m:7-13`), matching the
documentation's snap-to-voxel statement (`DIRandCOPD.md:6`).

Important nuance: the headline 4DCT row corresponds to the 1 mm, `cp_refinements=0`,
`loc_cc_approximate=true` table row (`DIRandCOPD.md:35-42`). That is the "fast LCC"
variant, not the exact-gradient LCC variant.

### COPDgene

The official COPD scripts always use the pTV 1 mm crop-resampling path.

| Item | pTVReg COPD behavior | Source |
|---|---|---|
| Fixed/moving | `get_COPD_data` returns eBH first and iBH second; the scripts assign these to moving and fixed, i.e. `iBH <- eBH` | `external/pTVreg/mutils/My/DIR_helpers/get_COPD_data.m:1`, `get_COPD_data.m:29-46`, `external/pTVreg/examples_ptv/COPD_final.m:7-12` |
| Intensity | same `[80,900] -> [0,1]` CT clamp | `COPD_final.m:10-12`, `COPD_finetune.m:11-13` |
| Crop | landmark bounding box, margin `[x,y,z]=[10,10,5]` | `COPD_final.m:14-24`, `COPD_finetune.m:15-25` |
| 1 mm resampling | target shape `round(crop_size .* spacing)`, spacing set to `[1,1,1]` | `COPD_final.m:26-31`, `COPD_finetune.m:27-33` |
| Test grid options | no refinement: `4`; refinement: start `8`, then refine once | `COPD_final.m:33-41` |
| Best/fine-tune grid | start `8 x 8 x 8`, `cp_refinements=1` | `COPD_finetune.m:35-37` |
| Best/fine-tune metric | `loc_cc_fftn_gpu`, sigma 2.1 mm, exact LCC gradient, isoTV 0.11 | `COPD_finetune.m:38-54` |
| Best/fine-tune interpolation | `interp_type=1` and `spline_order=1` | `COPD_finetune.m:46-50` |

The documentation states COPD uses 1 mm resampling, starts with `8 x 8 x 8`
control spacing and refines to `4 x 4 x 4`, LCC sigma 2.1 mm, and isoTV 0.11
(`external/pTVreg/DIRandCOPD.md:20-24`). The reported COPD best mean TRE is
`0.8461 mm` and mean time `442 sec` (`DIRandCOPD.md:9-12`).

The "Cubic interpolation" COPD row means cubic image interpolation/warping
(`interp_type=1`), not cubic control grids: the best script still sets
`spline_order=1` (`COPD_finetune.m:46-50`).

### Lung Masks

pTVReg's documentation explicitly highlights that its DIR result is among methods
that do not use lung masks (`external/pTVreg/Examples.md:124-125`) and that its COPD
result is the best among methods not using lung masks (`Examples.md:159-160`).

The example scripts do not pass explicit lung masks into `ptv_register`. They do set
`border_mask=5` (`DIR_test_all.m:76`, `COPD_finetune.m:44`). Internally, pTVReg
creates a 5-voxel border mask as `fixed_mask` if no fixed mask is supplied
(`external/pTVreg/ptv/ptv_process_input.m:155-167`) and passes it into the LCC cache
(`external/pTVreg/ptv/ptv_register.m:480-486`). The local-CC implementation uses that
mask through the fixed standard-deviation inverse terms, but does not simply multiply
the final LCC image by a stored hard mask because `cache.maskfix_pad` is set empty
(`external/pTVreg/mutils/My/image_metrics/create_loc_cc_fftn_cache.m:49-73`,
`external/pTVreg/mutils/My/image_metrics/metric_loc_cc_fftn.m:55-89`).

## pTVReg Objective and Regularization

### Similarity Loss

pTVReg's lung scripts use local cross-correlation, not global NCC:

| Detail | Source |
|---|---|
| Example scripts set `metric='loc_cc_fftn_gpu'` | `DIR_test_all.m:70`, `COPD_finetune.m:40` |
| Example scripts set LCC sigma as `metric_param=[1,1,1]*2.1` and `scale_metric_param=true` | `DIR_test_all.m:71-72`, `COPD_finetune.m:41-42` |
| At each pyramid level, sigma in pixels is `metric_param ./ cur_pix_resolution` | `external/pTVreg/ptv/ptv_register.m:346-352` |
| The LCC cache creates an FFT Gaussian kernel and precomputes fixed mean/std | `external/pTVreg/mutils/My/image_metrics/create_loc_cc_fftn_cache.m:15-47` |
| The metric computes local means/stds, local cross-product, `lcc=sprod/sgm`, and sums `(1-lcc)*dvol` | `external/pTVreg/mutils/My/image_metrics/metric_loc_cc_fftn.m:45-53`, `metric_loc_cc_fftn.m:87-113` |
| `loc_cc_approximate=true` uses an approximate gradient; false uses the exact formula | `metric_loc_cc_fftn.m:65-79` |

So "NCC" is not the same thing between pTVReg and our OASIS-style experiments:
pTVReg uses local CC with a 2.1 mm Gaussian kernel, while our SINR/DirectCP pTV-prep
runs use global NCC over the full registration crop.

### Regularization

The final lung scripts set only isoTV among the pTV spatial regularizers:

| Detail | Source |
|---|---|
| 4DCT sets `isoTV=0.11`, `csqrt=5e-3`, `spline_order=1` | `DIR_test_all.m:73-75` |
| COPD best sets `isoTV=0.11`, `csqrt=5e-3`, `spline_order=1` | `COPD_finetune.m:43-50` |
| Other regularizers default to zero unless explicitly set (`D1L2`, `D2L2`, `D2L1`, `D1L1`, etc.) | `external/pTVreg/ptv/ptv_process_input.m:77-101` |
| The default optimizer is `lbfgs` | `ptv_process_input.m:66-68` |
| The lung scripts set `max_iters=80` per level | `DIR_test_all.m:77`, `COPD_finetune.m:51` |
| pTVReg passes `MaxIter` and `Method` to minFunc | `external/pTVreg/ptv/ptv_register.m:533-539`, `ptv_register.m:608-638` |

Because `regularize_directly` defaults to false (`ptv_process_input.m:144-146`),
isoTV is applied to the control parameters, not directly to the dense displacement
field. The gradient spacing for this regularizer is `pix_resolution .* grid_spacing`
(`external/pTVreg/ptv/ptv_full_grad.m:151-157`). The isoTV penalty uses
`sqrt(sum(grad^2) + c)` with `c=csqrt` and integrates by the grid-cell volume
(`external/pTVreg/mutils/My/regularizers/isotropic_tv_regularization.m:1-4`,
`isotropic_tv_regularization.m:30-43`, `isotropic_tv_regularization.m:63-64`).

This matters for comparison: our OASIS-style SINR/DirectCP uses dense normalized
displacement L2, not pTVReg's physical-unit isoTV on control parameters.

## pTVReg Image Pyramid and Control Refinement

pTVReg has two different multiresolution mechanisms.

### 1. Image Pyramid

The lung scripts set `k_down=0.7` (`DIR_test_all.m:68`, `COPD_finetune.m:38`). If
`nlvl` is not supplied, pTVReg computes the number of image-pyramid levels from image
size, grid spacing, and `k_down` (`external/pTVreg/ptv/ptv_process_input.m:148-154`).

The pyramid is not just blur. pTVReg actually shrinks the image:

| Operation | Source |
|---|---|
| pTVReg creates pyramids for moving, fixed, and masks | `external/pTVreg/ptv/ptv_register.m:293-304` |
| Default pyramid interpolation is `interp_type=0`; lung calls pass `0` | `external/pTVreg/ptv/ptv_create_pyramid.m:10-23`, `ptv_register.m:293-304` |
| Next image size is `ceil(k_down * previous_size)` | `ptv_create_pyramid.m:56-62` |
| Voxel spacing is divided by `k_down` at each coarser level | `ptv_create_pyramid.m:62` |
| The image is Gaussian-smoothed and then resized with `volresize` | `ptv_create_pyramid.m:50-54`, `ptv_create_pyramid.m:73-80` |

For the lung setting `k_down=0.7` and `fine_pyramid=false`, the Gaussian sigma is
`0.4 * (0.5 / 0.7) ~= 0.286` pixels per axis (`ptv_create_pyramid.m:50-54`).

During the image-pyramid part, pTVReg keeps the nominal grid spacing equal to the
input `grid_spacing`; it does not halve the control spacing at each image level
(`external/pTVreg/ptv/ptv_register.m:317-339`). Because the image arrays shrink and
the pixel spacing increases, the number of knots and their physical spacing still
change across levels.

### 2. Control-Point Refinement

Only after all image-pyramid levels does `cp_refinements` add true control-grid
refinement. pTVReg loops over `nlvl + cp_refinements` (`ptv_register.m:317`). In the
refinement branch it halves `cur_grid_spacing` and upsamples the previous knots
(`ptv_register.m:340-343`).

The official lung scripts use:

| Dataset/config | Control refinement |
|---|---|
| 4DCT headline | `4 x 4 x 4`, `cp_refinements=0` |
| 4DCT optional refined 1 mm rows | `8 x 8 x 8 -> 4 x 4 x 4`, `cp_refinements=1` |
| 4DCT original-resolution rows | `4 x 4 x 3`, `cp_refinements=0` |
| COPD best/fine-tune | `8 x 8 x 8 -> 4 x 4 x 4`, `cp_refinements=1` |

This means pTVReg uses at most one control-grid refinement in the official lung
scripts, but it may use many image-pyramid levels before that refinement.

### How pTVReg Transfers Between Levels

For the official lung scripts, `spline_order=1`, so the control grid is linear. The
linear grid size is `ceil(image_size ./ grid_spacing) + 1`
(`external/pTVreg/ptv/ptv_get_grid_size.m:1-8`).

When moving to a new image-pyramid level or to a refined control spacing, pTVReg calls
`ptv_upsample_knots` (`external/pTVreg/ptv/ptv_register.m:338`,
`ptv_register.m:343`). For `K_ord==1`, that calls `refine_linear_grid_3d_v2` with
`'variation'` (`external/pTVreg/ptv/ptv_upsample_knots.m:9-22`).

The actual transfer is:

1. Generate the old dense displacement field from old knots:
   `linear_disp_3d(Knots, old_vsz, old_spacing)`.
2. Resize that dense field to the new image size with `volresize(..., new_vsz, 0)`.
3. Optimize new knots so their dense field matches the resized dense field.

These steps are implemented in
`external/pTVreg/mutils/My/deformation_tools_cpp/refine_linear_grid_3d_v2.m:19-35`.
The fitting objective is a squared dense-field difference, with gradients through the
linear control grid (`refine_linear_grid_3d_v2.m:38-45`).

Therefore pTVReg's transfer is dense-field resize plus variational projection into a
new linear control grid. It is not analytical knot insertion, and it is not an
explicit composition of multiple incremental transforms. Each level initializes an
absolute displacement parameterization from the previous solution and optimizes it.

## pTVReg Original Resolution vs 1 mm Resampled Results

pTVReg explicitly reports 4DCT original-resolution and 1 mm resampled rows. The docs
table is at `external/pTVreg/DIRandCOPD.md:35-42`; the script sweep is at
`external/pTVreg/examples_ptv/DIR_test_all.m:9-13`.

Values below are from the official saved artifact
`external/pTVreg/examples_ptv/results/DIR_TRE.mat` and match the docs up to rounding:

| pTVReg 4DCT config | Resolution | Mean TRE mm | Mean time sec | Per-case TRE mm |
|---|---|---:|---:|---|
| `cp_ref=0`, exact LCC | original crop | 0.962 | 74.9 | 0.824, 0.830, 0.977, 1.284, 1.126, 0.883, 0.827, 1.048, 0.946, 0.872 |
| `cp_ref=0`, fast LCC | original crop | 0.958 | 60.0 | 0.780, 0.791, 0.956, 1.269, 1.094, 0.888, 0.861, 1.069, 0.944, 0.929 |
| `cp_ref=0`, exact LCC | 1 mm crop | 0.930 | 191.5 | 0.804, 0.768, 0.921, 1.283, 1.110, 0.807, 0.796, 1.120, 0.905, 0.786 |
| `cp_ref=0`, fast LCC | 1 mm crop | 0.918 | 133.2 | 0.767, 0.747, 0.927, 1.259, 1.072, 0.834, 0.800, 1.014, 0.911, 0.844 |
| `cp_ref=1`, exact LCC | 1 mm crop only | 0.922 | 304.3 | 0.801, 0.768, 0.924, 1.296, 1.127, 0.784, 0.793, 0.996, 0.909, 0.820 |
| `cp_ref=1`, fast LCC | 1 mm crop only | 0.920 | 180.5 | 0.776, 0.743, 0.919, 1.272, 1.087, 0.837, 0.807, 0.990, 0.918, 0.848 |

For the clean `cp_ref=0` comparison, 1 mm resampling improves mean TRE by about
`0.032 mm` for exact LCC and `0.040 mm` for fast LCC, but increases runtime
substantially. This is a real protocol difference, not just a minor implementation
detail.

Important: pTVReg's "original resolution" rows are still thresholded, landmark-cropped,
no-lung-mask pTVReg runs. They are not equivalent to original IDIR, which uses a full
native image domain and lung-mask sampling in the audited reproduction protocol. See
`codex_auto_exp/IMPORTANT_LUNGCT_IDIR_PTVREG_PROTOCOL_BIAS.md` for that broader
comparison.

## Input Image Size and Protocol Comparison

The pTV 1 mm registration target shape is `round(crop_shape_zyx * native_spacing_zyx)`.
pTVReg implements the same formula in MATLAB after landmark crop
(`DIR_test_all.m:40-47`, `COPD_finetune.m:27-33`). Our pTV-prep runners use the same
shape formula, but with PyTorch trilinear interpolation rather than pTVReg's
`volresize(..., 1)` path (`codex_auto_exp/task4_lungct/scripts/run_sinr_oasis_cps4_ptvprep_lungct_case.py:131-134`,
`run_sinr_oasis_cps4_ptvprep_lungct_case.py:262-266`).

4DCT shapes below are `[z,y,x]` and were computed in the earlier protocol audit:

| Case | IDIR/native full shape | Native spacing mm | pTV crop shape | pTV 1 mm reg shape | Crop volume / native |
|---:|---:|---:|---:|---:|---:|
| 1 | `[94,256,256]` | `[2.5,0.97,0.97]` | `[84,155,239]` | `[210,150,232]` | 0.505 |
| 2 | `[112,256,256]` | `[2.5,1.16,1.16]` | `[101,168,241]` | `[252,195,280]` | 0.557 |
| 3 | `[104,256,256]` | `[2.5,1.15,1.15]` | `[96,151,236]` | `[240,174,271]` | 0.502 |
| 4 | `[99,256,256]` | `[2.5,1.13,1.13]` | `[91,159,216]` | `[228,180,244]` | 0.482 |
| 5 | `[106,256,256]` | `[2.5,1.10,1.10]` | `[91,165,231]` | `[228,182,254]` | 0.499 |
| 6 | `[128,512,512]` | `[2.5,0.97,0.97]` | `[92,166,270]` | `[230,161,262]` | 0.123 |
| 7 | `[136,512,512]` | `[2.5,0.97,0.97]` | `[87,184,284]` | `[218,178,275]` | 0.128 |
| 8 | `[128,512,512]` | `[2.5,0.97,0.97]` | `[91,213,262]` | `[228,207,254]` | 0.151 |
| 9 | `[128,512,512]` | `[2.5,0.97,0.97]` | `[66,186,233]` | `[165,180,226]` | 0.085 |
| 10 | `[120,512,512]` | `[2.5,0.97,0.97]` | `[84,215,216]` | `[210,209,210]` | 0.124 |

COPD pTV-style shapes:

| Case | Native full shape | Native spacing mm | pTV crop shape | pTV 1 mm reg shape | Crop volume / native |
|---:|---:|---:|---:|---:|---:|
| 1 | `[121,512,512]` | `[2.5,0.625,0.625]` | `[110,321,366]` | `[275,201,229]` | 0.407 |
| 2 | `[102,512,512]` | `[2.5,0.645,0.645]` | `[86,306,382]` | `[215,197,246]` | 0.376 |
| 3 | `[126,512,512]` | `[2.5,0.652,0.652]` | `[101,289,405]` | `[252,188,264]` | 0.358 |
| 4 | `[126,512,512]` | `[2.5,0.590,0.590]` | `[101,293,395]` | `[252,173,233]` | 0.354 |
| 5 | `[131,512,512]` | `[2.5,0.647,0.647]` | `[105,298,383]` | `[262,193,248]` | 0.349 |
| 6 | `[119,512,512]` | `[2.5,0.633,0.633]` | `[101,284,388]` | `[252,180,246]` | 0.357 |
| 7 | `[112,512,512]` | `[2.5,0.625,0.625]` | `[102,267,402]` | `[255,167,251]` | 0.373 |
| 8 | `[115,512,512]` | `[2.5,0.586,0.586]` | `[100,299,409]` | `[250,175,240]` | 0.406 |
| 9 | `[116,512,512]` | `[2.5,0.664,0.664]` | `[96,287,398]` | `[240,191,264]` | 0.361 |
| 10 | `[135,512,512]` | `[2.5,0.742,0.742]` | `[93,286,401]` | `[232,212,298]` | 0.301 |

Protocol implication: TRE is evaluated in mm on original landmarks for all methods,
but the optimization problem can still be biased by crop, resampling, mask/domain,
metric, and interpolation history. For cases 6-10, the pTV landmark crop is only about
8.5% to 15.1% of the native full 4DCT volume.

## Our Exact-Knot DirectCP Multistage Method

The current multistage runner is
`codex_auto_exp/task4_lungct/scripts/run_multistage_directcp_oasis_ptvprep_lungct_case.py`.

Important terminology clarification: in our runner, names like `16-8-4` are
full-resolution-equivalent control spacings. They are paired with an image
downsampling schedule. The actual control spacing on the stage image is
`domain_cps = full_cps / downsample` (`run_multistage_directcp_oasis_ptvprep_lungct_case.py:176-182`,
`run_multistage_directcp_oasis_ptvprep_lungct_case.py:210-225`,
`run_multistage_directcp_oasis_ptvprep_lungct_case.py:339-347`). Therefore:

| Our stage | Downsampled image scale | Full-resolution-equivalent spacing | Stage-image spacing |
|---|---:|---:|---:|
| stage 0 | `1/4` | `16` | `4` |
| stage 1 | `1/2` | `8` | `4` |
| stage 2 | `1` | `4` | `4` |

So our `16-8-4` / `4-2-1` setting was indeed intended as "cps4 on each image-pyramid
level" in the same broad sense as pTVReg's 4DCT `cp_refinements=0` setting. The
schedule name is from the final 1 mm image coordinate system, while pTVReg's
`grid_spacing=4` is from each current pyramid image coordinate system.

What it does:

| Item | Our implementation | Source |
|---|---|---|
| Objective | global NCC full crop plus `alpha *` SINR dense displacement L2 | `run_multistage_directcp_oasis_ptvprep_lungct_case.py:2-11`, `run_multistage_directcp_oasis_ptvprep_lungct_case.py:367-374` |
| Default schedule | `16,8,4` with downsample `4,2,1`, epochs `500,800,1200`, `lr=1e-3`, `alpha=100` | `run_multistage_directcp_oasis_ptvprep_lungct_case.py:112-121` |
| pTV-style input | same clamp, landmark crop, 1 mm shape formula, no lung-mask optimization | `run_multistage_directcp_oasis_ptvprep_lungct_case.py:531-546`, `run_multistage_directcp_oasis_ptvprep_lungct_case.py:569-584` |
| Stage pyramid | Gaussian blur with sigma `0.5 * downsample_factor`, then trilinear downsample | `run_multistage_directcp_oasis_ptvprep_lungct_case.py:185-197`, `run_multistage_directcp_oasis_ptvprep_lungct_case.py:339-347` |
| Control basis | direct cubic B-spline controls in normalized PyTorch `[x,y,z]` units | `run_multistage_directcp_oasis_ptvprep_lungct_case.py:701-704` |
| Stage transfer | exact crop-aligned cubic B-spline dyadic knot insertion | `run_multistage_directcp_oasis_ptvprep_lungct_case.py:272-306`, `run_multistage_directcp_oasis_ptvprep_lungct_case.py:468-483` |
| Exact insertion rule | separable crop-aligned cubic two-scale mask | `codex_auto_exp/task2_3_inr_work/scripts/nested_bspline_knot_utils.py:1-14`, `nested_bspline_knot_utils.py:67-101` |
| Trilinear resize comparison | diagnostic alternative only | `nested_bspline_knot_utils.py:104-107`, `run_multistage_directcp_oasis_ptvprep_lungct_case.py:283-305` |

Our pTV-prep helper uses `F.interpolate(..., mode="trilinear", align_corners=True)` for
image and displacement resizing (`run_sinr_oasis_cps4_ptvprep_lungct_case.py:131-140`).
pTVReg's MATLAB `volresize` uses the repository's interpolation flag path; `volresize`
defaults to `interp_type=1`, takes an explicit `interp_type` argument, and calls
`imdeform3` (`external/pTVreg/mutils/My/image_utils/volresize.m:1-7`,
`volresize.m:37-49`). The pTV helper convention maps `interp_type=0` to linear and
`interp_type=1` to cubic in the upsample matrix path
(`external/pTVreg/mutils/My/deformation_tools_cpp/create_3d_image_upsample_matrix.m:20-24`).

So our preprocessing is pTV-style and shape-matched, but not bitwise pTVReg
preprocessing.

## 4DCT DirectCP Multistage Results

From
`codex_auto_exp/task4_lungct/doc/lungct_oasis_multistage_directcp_4dct_all10_report.md`:

| Method | Cases | Mean snap TRE mm | Delta vs pTVReg | Delta vs IDIR repro | FG `J<=0` percent | Mean sec |
|---|---:|---:|---:|---:|---:|---:|
| pTVReg official artifact | 10 | 0.918 | 0.000 | -0.209 | n/a | n/a |
| IDIR reproduced | 10 | 1.127 | 0.209 | 0.000 | n/a | n/a |
| SINR cps4 lr1e-4 | 10 | 2.383 | 1.465 | 1.256 | 0.2594 | n/a |
| DirectCP cps4 lr1e-3 | 10 | 6.142 | 5.224 | 5.015 | 1.4234 | n/a |
| DirectCP multistage 8-4 | 10 | 2.581 | 1.664 | 1.455 | 0.3317 | 169.9 |
| DirectCP multistage 16-8-4 | 10 | 1.236 | 0.319 | 0.109 | 0.0520 | 203.5 |
| DirectCP multistage 8-4-2 | 10 | 2.881 | 1.963 | 1.754 | 0.3065 | 245.8 |
| DirectCP multistage 16-8-4-2 | 10 | 1.281 | 0.363 | 0.154 | 0.0480 | 303.6 |

The source report has the aggregate table at
`codex_auto_exp/task4_lungct/doc/lungct_oasis_multistage_directcp_4dct_all10_report.md:22-33`
and the schedule table at
`lungct_oasis_multistage_directcp_4dct_all10_report.md:13-20`.

The main readout there is:

1. `16-8-4` is the best exact-knot DirectCP multistage setting with mean snap TRE
   `1.236 mm`.
2. It reduces DirectCP cps4 from `6.142 mm` to `1.236 mm`, close to IDIR reproduced
   `1.127 mm`, but still above pTVReg official `0.918 mm`.
3. Starting at 8 mm is not enough for large-motion cases 7-8.
4. Adding a 2 mm final stage lowers image NCC loss but worsens landmark TRE.
5. Exact knot insertion preserved the dense field to max absolute difference
   `1.192e-07`; trilinear control resizing reached `5.649e-01`.

These readouts are in
`lungct_oasis_multistage_directcp_4dct_all10_report.md:50-56`.

## Follow-Up: Border Mask and pTV-Loss DirectCP

After the audit above, we ran two additional all-10 4DCT variants using the same
DirectCP `16-8-4` / downsample `4-2-1` schedule, epochs `500,800,1200`, Adam
`lr=1e-3`, and exact cubic knot insertion:

1. Border-mask global NCC: 5-voxel crop border mask, then global NCC plus the same
   `100 *` dense displacement L2 regularizer.
2. pTV-like loss: same border mask, Gaussian local CC with `sigma=2.1 mm`, plus isoTV
   control regularization with `weight=0.11` and `csqrt=5e-3`.

Full tables are in
`codex_auto_exp/task4_lungct/doc/lungct_directcp_16_8_4_border_ptvloss_4dct_all10_report.md`.

| Method | Cases | Mean snap TRE mm | Delta vs pTVReg | Delta vs IDIR repro | FG `J<=0` percent | Mean sec |
|---|---:|---:|---:|---:|---:|---:|
| pTVReg official artifact | 10 | 0.918 | 0.000 | -0.209 | n/a | n/a |
| IDIR reproduced | 10 | 1.127 | 0.209 | 0.000 | 0.0000 | 218.4 |
| DirectCP `16-8-4`, unmasked global NCC | 10 | 1.236 | 0.319 | 0.109 | 0.0520 | 203.5 |
| DirectCP `16-8-4` + border-mask global NCC | 10 | 1.176 | 0.259 | 0.049 | 0.0410 | 219.6 |
| DirectCP `16-8-4` + pTV LCC + isoTV | 10 | 1.661 | 0.743 | 0.534 | 0.0878 | 209.1 |

Readout:

1. The pTV-style border mask helps the old global-NCC multistage result slightly:
   `1.236 -> 1.176 mm` mean TRE, with similar log-J and slightly lower folding.
2. The pTV-like LCC+isoTV variant improves many individual cases, but it fails case 8
   (`7.391 mm` vs `1.549 mm` for border-mask global NCC and `1.014 mm` for official
   pTVReg). This dominates the mean.
3. Therefore, this run should not be read as "pTVReg's loss is worse." It is still not
   exact pTVReg: it keeps our cubic controls, exact knot insertion, dyadic pyramid,
   Adam optimizer, and PyTorch local-CC approximation rather than pTVReg's linear grid,
   `k_down=0.7` pyramid, minFunc/L-BFGS, and MATLAB LCC/cache behavior.
4. The cleanest conclusion is that `border_mask=5` is worth keeping for our global-NCC
   schedule, while a faithful pTVReg-style DirectCP reproduction needs more of the
   pTVReg optimization stack before drawing conclusions about LCC+isoTV.

## What We Match vs Do Not Match

| Aspect | pTVReg official lung | Our exact-knot DirectCP multistage | Match status |
|---|---|---|---|
| CT intensity | `[80,900] -> [0,1]` | same formula | matched |
| Landmark crop | `[x,y,z]=[10,10,5]` | same margin, zyx implementation | matched |
| 1 mm target shape | `round(crop_shape * native_spacing)` | same shape formula | mostly matched |
| 1 mm interpolation | MATLAB `volresize(...,1)` path | PyTorch trilinear `align_corners=True` | not exact |
| Lung mask | no lung mask | no lung mask for optimization | matched |
| Border mask | `border_mask=5` in pTV cache | full crop global NCC, no pTV border mask | not matched |
| Similarity | local CC, sigma 2.1 mm | global NCC over full crop | not matched |
| Regularizer | isoTV on control grid in physical units | dense normalized displacement L2 | not matched |
| Control basis | linear displacement grid (`spline_order=1`) | cubic B-spline controls | not matched |
| Optimizer | minFunc/L-BFGS default, 80 iterations per level | Adam, thousands of epochs | not matched |
| Image pyramid | many levels, `k_down=0.7`, shrink by `ceil(0.7 * size)` | hand schedules like `4,2,1` downsample | not matched |
| CP refinement | 4DCT headline: no extra CP refinement; nominal cps4 on every image-pyramid level. COPD best: nominal cps8 through the pyramid, then cps4 refinement | `16->8->4` with downsample `4->2->1` is cps4 on each stage image; `*-2` schedules add a true cps2 final stage | partly matched |
| Level transfer | dense-field resize + variational projection | analytical cubic knot insertion | intentionally different |
| TRE convention | original landmarks after resize-back/uncrop; pTV rounds moved points before TRE | original landmarks after resize-back/uncrop, snap headline | mostly matched |

## Interpretation

The DirectCP multistage experiments show that capture range is a major failure mode:
single-stage DirectCP cps4 fails badly on lung, while `16-8-4` brings mean TRE down to
`1.236 mm`. This is strong evidence that coarse-to-fine initialization matters.

But the remaining pTVReg gap should not be attributed only to knot insertion or CP
spacing. pTVReg also differs in the similarity metric, regularizer, control basis,
optimizer, image pyramid, and border-mask/cache behavior. The largest methodological
mismatch is likely local CC plus isoTV plus the pTV pyramid, not just whether the
control field is initialized well.

The pTVReg 4DCT headline result is especially easy to misread. It says `4 x 4 x 4`
control spacing, but the code optimizes over many image-pyramid levels before reaching
the final 1 mm crop. So it is more like "same nominal 4-pixel grid over a pTV image
pyramid" than "one direct cps4 solve on the full 1 mm crop."

For COPD, pTVReg's best result is closer in spirit to our `8-4` schedules because it
uses one final control refinement from 8 to 4. However, it still uses pTV's local CC,
isoTV, linear grid, and L-BFGS machinery, not our global NCC + dense L2 + cubic
exact-knot DirectCP.

## Recommended Next Experiments

To understand the pTVReg gap cleanly, I would not jump straight to more knot schedules.
The next step should be a pTV-style ablation ladder:

1. Implement a PyTorch pTV-like DirectCP baseline on the same pTV-prep crop:
   local CC with sigma `2.1 mm`, `border_mask=5` behavior, isoTV on control points in
   physical units, linear displacement grid, L-BFGS or PyTorch LBFGS, and pTV-like
   `k_down=0.7` image pyramid.
2. Reproduce the official pTVReg settings before modifying them:
   4DCT `grid=4`, `cp_ref=0`; COPD `grid=8`, `cp_ref=1`.
3. Add one factor at a time:
   local CC only, then isoTV, then pTV image pyramid, then optimizer, then linear vs
   cubic control basis.
4. Compare pTVReg variational transfer against our exact cubic knot insertion as an
   explicit ablation. Exact insertion is mathematically cleaner for our cubic nested
   basis, but pTVReg's variational projection may interact differently with linear
   grids and resized images.
5. Keep protocol labels in every table:
   `native_full_lungmask_idir`, `ptv_landmark_crop_1mm_no_lungmask`,
   `ptv_landmark_crop_1mm_global_ncc`, and so on.

The most actionable hypothesis is:

> Our `16-8-4` exact-knot DirectCP fixes much of the capture-range problem, but to close
> the remaining gap to pTVReg we probably need pTVReg's local CC + isoTV + image-pyramid
> recipe before spending more time on additional CP schedules.
