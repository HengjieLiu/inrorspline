# OASIS MR-SINR 8/4/2 Revision Plan

## Goal

Implement and run a minimal OASIS multiresolution SINR pilot before scaling to
larger reviewer-facing sweeps. The pilot compares the intended composed-field
MR-SINR design against a simpler continued/re-queried SINR design on one
validation case.

The pilot must answer whether the composed SINR schedule is technically stable
and worth promoting to the seed-3 first10 and first100 Pareto runs.

## Fixed Protocol

- Dataset: OASIS BrainMR.
- Split file: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Pilot case: seed-3 rank `0`.
- Pilot pair: `(0333, 0115)`.
- Expected original sorted `pair_test_200` index: `158`.
- Registration direction and pair semantics: use the existing OASIS loader and
  method conventions in `scripts/experiments/oasis_alpha_sweep/run_one.py`.
- Foreground convention for summaries and foreground-scoped regularity:
  fixed raw image-positive foreground, `fixed_image > 0`.
- Do not use `seg > 0` as the sampling/domain foreground. Segmentation labels
  are only for label-overlap metrics and visual/comparison audits.

## Methods To Implement

### `mr_sinr842_composed`

This is the primary intended reviewer-response method.

- Use three independent SINR B-spline-control stages with BSCP schedule
  `8,4,2`.
- Stage 8 trains on the original moving image.
- Stage 4 trains on the moving image already warped by the frozen stage-8
  field.
- Stage 2 trains on the moving image already warped by the frozen composed
  stage-8/stage-4 field.
- Final evaluation must compose pull fields back to the original moving image.
- Do not evaluate only the final staged warped image as the final deformation.
- Do not use naive displacement addition as the final deformation.

Pull-field composition convention:

```text
phi8(x) = x + u8(x)
phi4(x) = x + u4(x)
phi2(x) = x + u2(x)

phi84(x)  = phi8(phi4(x))
phi842(x) = phi8(phi4(phi2(x)))
```

Equivalent dense displacement form for two stages:

```text
u84(x) = sample(phi8, x + u4(x)) - x
       = u4(x) + sample(u8, x + u4(x))
```

The third stage follows the same pull-field composition rule. The code should
use the repository's existing `grid_sample(..., align_corners=True)` convention.

### `mr_sinr842_requery`

This is the implementation ablation.

- Use one continued SINR generator across the same BSCP schedule `8,4,2`.
- Re-query the current SINR at the finer B-spline-control grid at each stage.
- Optimize an absolute field at each stage.
- Do not compose fields between stages.
- Record transition diagnostics because this variant is not exact
  knot-insertion continuation.

## Optimization Defaults

- Total budget: `2500` epochs.
- Stage epochs: `400,800,1300`.
- Stage downsample factors: `4,2,1`.
- BSCP schedule: `8,4,2`.
- Keep SINR omega fixed at the existing default: `50.0`.
- Use the existing non-DirectCP default learning rate unless a method-specific
  default is explicitly added. Current default: `1e-4`.
- Use the existing OASIS SINR objective family:
  full-image global NCC plus analytic cubic B-spline bending regularization.
- First pilot uses one representative alpha. Prefer the central/main OASIS
  spline alpha already used for comparable figures.
- Do not tune omega in this revision branch.

## Required Outputs

For each pilot method, write normal per-run outputs plus a compact pilot
comparison summary under this experiment folder or the corresponding run root.
Include:

- Dice.
- HD95.
- ASD.
- Final image NCC.
- Bending energy.
- Diffusion.
- log-Jacobian standard deviation.
- Nonpositive-Jacobian fraction or percent, `J <= 0`.
- Training runtime and peak memory.
- Stage trace with stage BSCP, downsample, epochs, LR, and alpha.
- For `mr_sinr842_composed`, a composed-vs-sequential warp consistency
  diagnostic.
- For `mr_sinr842_requery`, a stage-transition jump diagnostic.

## Pilot Acceptance And Decision Rule

The pilot passes technically only if both methods produce finite losses and
metrics, output the expected deformation/summary artifacts, and pass the
orientation/composition consistency checks.

Promote `mr_sinr842_composed` to the seed-3 first10 and later first100 Pareto
work only if it is stable and not clearly worse than `mr_sinr842_requery` on
accuracy or regularity. Keep `mr_sinr842_requery` as an ablation only if it is
unexpectedly strong or helps explain the result.

Do not launch first10 or first100 runs until this one-case pilot has been
inspected.

## Scale-Up After Pilot

If the composed method is selected:

- First scale to OASIS seed-3 first10.
- Then run the first100 Pareto sweep with the selected alpha list.
- Keep the total 2500-epoch budget.
- Keep omega fixed.
- Preserve the seed-3 shuffled split and `image > 0` foreground convention.

## Notes For The Execution Agent

- Existing DirectCP multistage knot insertion is a different mechanism and
  should remain unchanged.
- MR-D-BSCP uses exact dyadic knot insertion for explicit controls.
- The composed MR-SINR method is not knot insertion; it is sequential residual
  registration with pull-field composition.
- Raw alpha values should only be compared within matched objective families.
