---
doc_type: literature_note
status: active_draft
---

# Zeybekoglu et al. 2026: Accessible pTVReg Python

## Bibliographic Anchor

- Title: An Accessible Solution for Deformable Image Registration Compared with
  Learning-Based Approaches.
- Authors: Onur Ali Zeybekoglu, David Tilly, and Orcun Goksel.
- Version reviewed: arXiv `2608.02248v1`, submitted August 3, 2026.
- Local source checked: `docs/literature/2608.02248v1.pdf`.
- Code checked: `external/ptvreg-python`.

## Relevant Method Details

- The paper positions pTVReg as an accessible variational registration baseline
  against learning-based methods.
- The public Python code exposes a minimal registration function,
  `register_pair_min(moving, fixed, opts)`, under `src/ptv`.
- The COPD-relevant reported setting uses pTV-style local cross-correlation and
  isotropic total-variation regularization with ADMM optimization.
- The paper describes HU clipping to `[-1000, 500]` followed by normalization
  to `[0, 1]` for CT inputs.
- The reviewed PDF does not by itself spell out the exact COPD crop, lung-mask,
  1 mm resampling, or landmark-coordinate conversion used for the Python
  experiments. The public Python repo also does not include a COPD
  preprocessing script.
- The fixed COPD experiment in this repo therefore tests staged, explicitly
  tagged inputs: canonical local `pTVReg_iso1mm`, an experiment-local paper-HU
  1 mm variant, and a raw-COPD native-crop paper-HU diagnostic gate.

## Interpretation Notes

- Reported paper values are not automatically method-only comparable with this
  repo's LungCT rows unless crop, spacing, interpolation, landmark evaluation,
  and regularity conventions are matched or explicitly tagged.
- The repo headline metric for these trials remains
  `native_crop_tre_snap_to_voxel` in mm, with subvoxel and cropped-1mm TRE
  retained as companion diagnostics where the registration grid is 1 mm. For
  native-crop runs, the registration-grid TRE is the native-crop TRE.
- Dense-field regularity is an inrorspline posthoc diagnostic and should not be
  read as a metric reported by the paper unless separately verified.
