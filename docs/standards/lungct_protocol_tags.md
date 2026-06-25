---
doc_type: lungct_protocol_tags
status: active
---

# LungCT Protocol Tags

LungCT/4DCT/COPD comparisons are protocol-sensitive. Before interpreting any
row, read:

- `docs/reports/protocol_warnings/lungct_idir_ptvreg_protocol_bias.md`
- `docs/reports/protocol_warnings/ptvreg_technique_audit_directcp_multistage.md`

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
