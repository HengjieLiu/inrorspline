---
doc_type: technical_log
status: active
---

# ptvreg-python COPD Fixed All10 Technical Log

## Known Limitations

- The canonical protocol uses the repository's regenerated COPD pTVReg-style
  1 mm preprocessing, whose audit records a legacy-Python crop convention that
  can differ by one voxel from the exact MATLAB pTVReg formula.
- The paper-HU protocol changes intensity preprocessing only; it intentionally
  reuses canonical crop bounds, masks, and landmarks to avoid mixing crop and
  intensity effects.
- The native paper-HU protocol is a diagnostic gate, not a claim of exact paper
  reproduction: it uses raw COPD volumes, exact pTVReg-style landmark crop
  bounds, paper-HU clipping/normalization, and raw COPD native spacing.
- The public Python implementation is imported as-is from a Git submodule and
  may not be numerically equivalent to the older MATLAB pTVReg implementation.
- Dense-field regularity is computed post hoc with the inrorspline foreground
  fixed-lung-mask metric convention; pTVReg itself is run without a lung-mask
  optimization domain.
- Native-grid dense-field regularity uses voxel finite differences on the
  anisotropic native crop and is tagged as such in summaries.

## 2026-08-28 Setup

- Added `external/ptvreg-python` as an isolated submodule commit:
  `8204459 Add external submodule ptvreg-python`.
- Submodule revision recorded at setup:
  `64ed3f2fd3bb4a5f5df1f0a165d521b9d1552715`.
- Adapter package:
  `scripts/experiments/ptvreg_python_repro/`.
- Planned run root:
  `/database/inrorspline/runs/ptvreg_python_copd_fixed_all10_v1`.

## 2026-08-28 Preflight

- Adapter tests passed in Docker; after adding the native protocol this is
  `12 passed in 1.42s`.
- Canonical COPD loader audit passed for cases `01-10`; saved as
  `canonical_loader_audit.json`.
- Visible Docker GPUs: four NVIDIA RTX 6000 Ada Generation GPUs, each
  reporting `49140 MiB`.
- Canonical four-GPU dry-run queued all ten cases and assigned cases
  round-robin across GPUs `0,1,2,3`.
- Tiny synthetic CUDA smoke passed with finite dense fields, shape
  `[20,24,28]`, no NaNs/infinities, and external pTVReg Python SHA
  `64ed3f2fd3bb4a5f5df1f0a165d521b9d1552715`; saved as
  `synthetic_gpu_smoke.json`.
- Full-volume canonical case01 reduced real-case smoke was attempted twice.
  The first used `max_iters=1`, `admm_max_outer=1`, `admm_lbfgs_iters=3`,
  `nlvl=1`; the second used `max_iters=1`, `admm_max_outer=1`,
  `admm_lbfgs_iters=1`, `nlvl=1`. Both remained inside the external
  registration call with no completed summary or log output and were manually
  stopped. GPU memory was released after stopping.
- Production all-10 registration is therefore gated pending an explicit
  compute/runtime decision or deeper profiling of the external ADMM/tricubic
  path. This is a runtime gate, not a field-conversion failure.

## 2026-08-28 Paper-HU Inputs

- Prepared all ten paper-HU COPD 1 mm fixed/moving input pairs under
  `/database/inrorspline/runs/ptvreg_python_copd_fixed_all10_v1/_paper_hu_iso1mm_inputs`.
- Derived input size on disk: approximately `896M`.
- Paper-HU input manifest saved as `paper_hu_input_manifest.json`; all ten
  cases reported `verification_pass=true`.
- Paper-HU four-GPU dry-run queued all ten cases and assigned cases
  round-robin across GPUs `0,1,2,3`.
- Placeholder aggregate reports were generated for both canonical and paper-HU
  protocols with `allow_missing`; both correctly report `0/10` completed and
  acceptance `false` because production registration was not launched.

## 2026-08-28 Native Paper-HU Case08 Gate

- Added `paper_hu_native` protocol:
  `ptvreg_python_admm+paper_hu_native_landmark_crop_no_lungmask`.
- Case08 native input cache was prepared from raw COPD under
  `/database/inrorspline/runs/ptvreg_python_copd_fixed_all10_v1/_paper_hu_native_inputs`.
- Case08 native crop: shape `[100,299,409]`, spacing `[2.5,0.586,0.586]` mm,
  crop bounds `[[6,105],[113,411],[54,462]]`, verification passed.
- Case08 fixed-options registration completed on GPU 0.
- Result: headline native-crop snap TRE `2.8865 mm`; native-crop subvoxel TRE
  `3.0143 mm`; identity TRE `26.4569 mm`.
- Dense regularity: foreground stdLogJ `0.5725`, foreground folding
  `2.4881%`, bending energy `0.060620`, diffusion L2 `0.090000`.
- Runtime: `914.5 s`; peak GPU memory: `46.72 GiB`.
- Operational note: peak memory leaves little margin on a 49 GiB RTX 6000 Ada.
  Do not start a production four-GPU all-10 run from this protocol without an
  explicit memory/risk decision; case sizes other than case08 may exceed the
  memory ceiling.
- Native all-10 dry-run manifest generated with GPUs `0,1,2,3`; case08 is
  skipped by default because it already completed, leaving nine queued jobs.
