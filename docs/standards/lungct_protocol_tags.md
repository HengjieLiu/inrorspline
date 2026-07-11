---
doc_type: lungct_protocol_tags
status: active
---

# LungCT Protocol Tags

LungCT/4DCT/COPD comparisons are protocol-sensitive. Before interpreting any
row, read:

- `docs/reports/protocol_warnings/lungct_idir_ptvreg_protocol_bias.md`
- `docs/reports/protocol_warnings/ptvreg_technique_audit_directcp_multistage.md`
- `docs/standards/lungct_tre_metric_conventions.md`

## Fixed/Moving Image Convention

This repository follows the pTVReg/DIR-Lab convention for LungCT registration.

### Registration Direction

For DIR-Lab 4DCT experiments, registration is performed from the end-exhale
phase to the end-inhale phase:

```text
moving image: T50 / end-exhale / maximum exhale
fixed image:  T00 / end-inhale / maximum inhale
direction:    T50 -> T00, i.e., exhale-to-inhale
```

In standard image registration terminology:

- The fixed image is the reference or target image space.
- The moving image is the image that is spatially transformed.
- The registration output is the moving image warped into the fixed image
  space.

Therefore, in the LungCT setup:

```text
moving image = exhale CT = smaller/contracted lung state
fixed image  = inhale CT = larger/expanded lung state
```

The registration task is to deform the smaller-lung exhale image toward the
larger-lung inhale image.

### DIR-Lab 4DCT Phase Naming

For the DIR-Lab 4DCT dataset:

```text
T00 = maximum inhale phase
T50 = maximum exhale phase
```

Accordingly:

```text
T00 has the larger lung volume.
T50 has the smaller lung volume.
```

When using the pTVReg-style setup, the images are assigned as:

```text
volmov = T50  # moving image, maximum exhale
volfix = T00  # fixed image, maximum inhale
```

Thus, the image registration direction is:

```text
T50 -> T00
exhale -> inhale
moving -> fixed
smaller lung -> larger lung
```

### COPD Breath-Hold CT Convention

For COPD experiments, the same fixed/moving interpretation is used:

```text
moving image: eBHCT / end-exhale breath-hold CT
fixed image:  iBHCT / end-inhale breath-hold CT
direction:    eBHCT -> iBHCT, i.e., exhale-to-inhale
```

Therefore:

```text
eBHCT = moving image = smaller/contracted lung
iBHCT = fixed image  = larger/expanded lung
```

### Displacement Field Direction

The image registration direction and the displacement-field sampling convention
should not be confused.

At the image level, the repository convention is:

```text
moving image is warped to fixed image
T50 / exhale -> T00 / inhale
```

However, depending on the implementation, the stored displacement field may be
represented as a backward/pull field rather than a forward physical trajectory
field. In a pull-based warping convention, each voxel in the fixed image grid
samples an intensity from a corresponding location in the moving image.
Therefore, the displacement array may numerically map fixed-grid coordinates
back into the moving image, even though the registration task is described as
moving-to-fixed.

For clarity in this repository:

```text
"registration direction" refers to the image-level task:
moving image -> fixed image

"field direction" refers to how the displacement field is stored or sampled
internally.
```

Unless otherwise stated, exhale-to-inhale LungCT registration means:

```text
moving image: exhale CT / T50 / eBHCT
fixed image:  inhale CT / T00 / iBHCT
warped image: moving exhale CT resampled into fixed inhale CT space
```

### Recommended Wording

A concise description for manuscripts or experiment summaries is:

> For LungCT registration, we follow the pTVReg/DIR-Lab convention and register
> the end-exhale image to the end-inhale image. Specifically, T50/end-exhale is
> used as the moving image and T00/end-inhale is used as the fixed image. Thus,
> the moving image corresponds to the smaller, contracted lung state, while the
> fixed image corresponds to the larger, expanded lung state.

A shorter version is:

> LungCT registration is performed in the exhale-to-inhale direction, with
> T50/end-exhale as the moving image and T00/end-inhale as the fixed image.

## Required Tag Fields

Every LungCT run or result row should make these fields explicit:

```text
dataset: 4dct | copd
case_set: case01...case10 | selected cases
image_domain: native_full_volume | landmark_crop
spacing: native | isotropic_1mm | other
crop_rule: none | landmark_bbox_margin_xyz_10_10_5 | other
intensity_rule: native | ct_clip_80_900_scale_0_1 | other
mask_rule: none | fixed_t00_lung_mask | other
loss_domain: full_image | crop | mask_sampled | border_masked
metric_domain: native_landmarks | crop_landmarks | other
metric_resolution: native_full_or_crop | cropped_1mm_reg | other
tre_mode: snap_to_voxel | subvoxel | both
method_family: idir | sinr | directcp | ptvreg | other
variant: method-specific short name
```

For IDIR-style LungCT rows, `fixed_t00_lung_mask` means the fixed/inspiration
`T00` lung mask only. The original/native IDIR convention samples training
coordinates from the fixed-image mask and does not use a moving/expiration `T50`
mask for optimization. Moving masks may still exist for QA, paired-mask audits,
or explicitly tagged non-IDIR variants.

## Standard Protocol Handles

Use these handles unless a run needs a more specific one:

- `native_full_lungmask_idir`: native full-volume IDIR-style run with fixed/T00
  lung-mask sampling. This handle uses only the fixed/T00 mask for optimization;
  any moving/T50 mask is QA-only unless a variant explicitly says otherwise.
- `ptv_landmark_crop_1mm_no_lungmask`: pTVReg-style landmark crop, 1 mm
  resampling, CT clipping/scaling, no lung-mask optimization.
- `ptv_landmark_crop_1mm_with_lungmask`: pTV-style crop and resampling with an
  explicit lung-mask loss or sampling convention.
- `ptv_directcp_multistage`: DirectCP run using pTV-style preprocessing and a
  multistage control-spacing schedule.
- `ptv_sinr_cps4`: SINR cps4 run using pTV-style preprocessing.
- `isoptv_torch_ptv_domain`: isoTV Torch run on the pTV-style domain.

## Current pTV-Prep Status

Legacy task4 pTV-prep and pTV-domain outputs are audit inputs, not canonical
processed data. Do not launch new claims from copied legacy pTV-prep,
DirectCP/SINR pTV-domain, or `isoptv_torch` outputs.

The current regenerated 4DCT pTV-prep handle is
`ptvreg_landmark_crop_1mm_trilinear_v1`, stored under the processed LungCT data
tree at `processed/preprocessing/ptvprep/`. It uses pTVReg fixed/moving,
intensity, crop-index, and target-shape conventions, with intentional PyTorch
trilinear image resampling. The transformed T00/T50 lung masks in this artifact
are QA/preprocessed-data companions, not evidence that official pTVReg used lung
masks for optimization.

## Interpretation Rule

Do not treat native IDIR, pTVReg, pTV-preprocessed SINR, and pTV-preprocessed
DirectCP rows as clean method-only comparisons unless preprocessing, crop, mask,
metric, interpolation, regularization scale, and evaluation convention are
matched or explicitly tagged.

For LungCT pTVReg-style comparisons, use `native_full_or_crop` +
`snap_to_voxel` as the headline TRE convention. If a method reports
`cropped_1mm_reg` TRE directly, label it as a 1mm registration-grid diagnostic
because transformed landmarks are usually fractional in that grid.
