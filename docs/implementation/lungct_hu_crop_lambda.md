# MR-D-BSCP HU, crop and lambda ablation

The P90 runner calls the existing `train_multistage` numerical implementation
without modifying its loss, optimizer, spline basis or prolongation. Every
candidate starts a fresh process and completes2500 steps. Lung masks define C1
and evaluation domains; they do not weight the registration objective.

H0/H1/H2 clip native HU to[-944,-124],[-1000,1000],[-1000,2000], respectively.
All use the same affine scale `(clippedHU+944)/820`, with no further clipping.
H0 uses the original reciprocal-multiplication implementation for bitwise input
parity. C0 uses corrected landmark-union bounds; C1 uses both phase lung masks
with physical padding; C2 uses the acquired image. Endpoint-aligned resizing and
native snap TRE are preserved, including the272-slice corrected COPD01 C0 grid.

Each case/preparation searches lambda0.055*2^k, k=-8..8. Upper and lower
branches compare successive accepted TRE values independently, stop on two
strict increases, and reset their streak after a tie, improvement or failed run.
A failure removes the previous comparison anchor; it is not an infinite score.
Caps and untested values are explicit. Selecting the best of the same300
landmarks is in-sample tuning, not independent validation.

The controller records immutable outcomes, checks free GPU memory against
reduced-run profiles, limits itself to one worker per GPU, and reports the first
four cases before automatically releasing the remaining16. Reports distinguish
fixed0.055 from individually tuned results and expose incomplete cohorts.

Fields remain FP32 ZYX3 pull maps in registration voxels; native copies and
per-landmark results use the audited P87 evaluator. Centered finite-difference
Jacobian summaries are tagged separately from legacy stencils. Independent
TotalSegmentator labels provide provisional bone/soft-tissue overlap and surface
metrics on common native crop coverage; labels never influence lambda selection.

Anatomy labels use pinned TotalSegmentator2.16.0, task `total`, `fast=False`
(1.5mm model), independently on each original raw-HU phase and restored to native
geometry. Bone is the union of ribs, vertebrae and sternum; heart/aorta/esophagus
retain separate labels. Lung metrics reuse the existing lungmask R231 masks.
The moving masks are pull-warped with nearest-neighbor interpolation. Labels are
not re-inferred for HU/crop conditions or warped images. All anatomy comparisons
use the case-specific C0/C1/C2 intersection; C2 therefore does not imply a
whole-FOV anatomy evaluation. Large-organ Dice does not establish internal
landmark accuracy, and union-bone Dice does not establish per-bone rigidity.

`summarize_anatomy_tradeoffs.py` separates fixed0.055 from TRE-selected
candidates, exports paired case changes and sample SDs, and plots their different
tradeoffs. These are descriptive repeated measurements from20 subjects; lambda
selection is in-sample. Folding diagnostics retain their original registration
domain, separate from the common-ROI anatomy measurements.

Bone-component reports retain combined-bone Dice and separately expose grouped
ribs, grouped vertebrae and sternum Dice for both fixed and TRE-selected lambda.
Combined-bone Dice is taken from the actual union mask, not the arithmetic
average of component Dice. Case tables and unweighted ten-case mean/sample-SD
tables retain counts; CSV/JSON use fractions and dedicated Markdown tables use
percent. Versioned supplements preserve the original sealed reports. The
analysis additionally exports component-specific paired HU/crop changes and
TRE-versus-overlap plots without rerunning registration or segmentation.

cuDNN benchmarkFalse/deterministicTrue is requested and verified, but does not
guarantee bitwise CUDA reproducibility. Repeated original-CLI reduced runs also
change locally; the external preflight records this limitation. Shared-device
timings are not isolated performance benchmarks.

Source: `scripts/experiments/lungct_hu_crop_lambda/`; specification and technical
log: `experiments/2026-11-19_lungct_hu_crop_lambda/`. Generated preparations,
frozen source, logs, fields, validation and reports remain under ARTIFACT_ROOT.
