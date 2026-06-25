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

## Standard Protocol Handles

Use these handles unless a run needs a more specific one:

- `native_full_lungmask_idir`: native full-volume IDIR-style run with fixed/T00
  lung-mask sampling.
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
DirectCP/SINR pTV-domain, or `isoptv_torch` outputs until the preprocessing
audit is complete and the preprocessing is regenerated under the current
protocol tags.

## Interpretation Rule

Do not treat native IDIR, pTVReg, pTV-preprocessed SINR, and pTV-preprocessed
DirectCP rows as clean method-only comparisons unless preprocessing, crop, mask,
metric, interpolation, regularization scale, and evaluation convention are
matched or explicitly tagged.
