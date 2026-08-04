# Technical log: COPD DUAL-INR six-arm baseline

## Predeclared limitations

- The current COPD 1 mm artifact is intentionally reused because it matches
  the existing MR-D-BSCP P2e cohort. Its crop uses legacy Python bounds, not
  exact official pTVReg MATLAB bounds.
- The official pTVReg COPD result has casewise TRE but no accessible dense
  displacement field, so regularity is reported as N/A.
- A0/A1 use a native author lobe-union crop; N uses native full volume; P uses
  a pTV-style 1 mm crop. These are not method-only rows.
- There is one seed and no COPD outcome-driven hyperparameter tuning.
- Diffusion and bending are finite-difference voxel-coordinate diagnostics and
  are protocol-tagged across native and 1 mm grids.

## Preflight

- Canonical COPD images, landmarks, fixed iBH R231 masks, and current pTV
  artifacts validate for cases 01-10.
- The intended Docker environment has TotalSegmentator 2.13.0 and cached
  five-lobe model components required by the author contract.

## Review-artifact portability

- Before commit, machine-local paths in the compact tracker and resolved-plan
  artifacts were normalized to symbolic run, review, and input roots. This
  serialization-only change did not alter TRE, regularity, or acceptance data.
