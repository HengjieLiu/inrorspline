# 1.5-mm COPD P2e corrected-retiming handoff

## Purpose and scope

This handoff specifies a **corrected, timing-oriented rerun** of the completed
MR-D-BSCP P2e adaptation on the challenge-like COPD cohort. It freezes the
completed registration protocol and removes only a redundant diagnostic
operation: non-final full-grid DVF expansion after a stage completes.

The baseline rerun is limited to COPD cases `104` through `113` with the
historical-P2e-equivalent intensity profile (`p2e_equiv_hu`). It is not a
two-window comparison, validation cohort run, submission, or acceleration
study. The OASIS acceleration strategies below are adaptation references only;
they must not silently change this baseline rerun.

## Source implementation to copy and pin

The implementation used for the completed 1.5-mm cohort is currently
untracked in this repository. Copy and version the source files themselves;
the enclosing Git commit alone is not sufficient provenance.

| Item | Value |
| --- | --- |
| Repository commit at handoff | `97e18208a36fa3699c94e23f8735a8103fa696a4` |
| P2e runner | `scripts/experiments/learn2breath_mrdbscp_p2e/run_pairlocal.py` |
| Runner SHA-256 | `6396aa369a84af81a78e95dbdf5204e1a343e42124341e07d2c9ca74bf2c0de6` |
| Original four-lane controller | `scripts/experiments/learn2breath_mrdbscp_p2e/run_all_two_window_audit.py` |
| Controller SHA-256 | `32c0cc6726e98448b849391a79a776547c2628e64c70008b38d6552e311585e0` |

The original controller is a **40-job, four-GPU** audit controller. Do not
invoke it unchanged for this rerun. Reuse its manifest, persistent-worker,
cold/warm, and aggregation patterns to create a single ten-job COPD P2e lane.

Use portable environment variables in the new repository; do not copy private
host paths into code or reports:

```text
COPD_PAIRLOCAL_ROOT   # challenge-like COPD NIfTI/landmark product
R231_ROOT             # optional geometry/QC masks only
L2B_EVALUATOR_ROOT    # repository that provides challenge folding evaluation
RUNTIME_ROOT          # external output location, never Git
```

## Frozen registration protocol

### Inputs, direction, and field contract

- Cohort: cases `104,105,106,107,108,109,110,111,112,113`.
- Fixed image: `INSP` (`case_<id>_1.nii.gz`); moving image: `EXP`
  (`case_<id>_2.nii.gz`). Registration direction is EXP -> INSP.
- Every fixed/moving pair must be a finite, affine-matched NIfTI volume of
  shape `[X,Y,Z] = [224,192,224]` at 1.5-mm isotropic spacing.
- The internal historical P2e solver uses `[Z,Y,X]` tensors. Its exported DVF
  is `float32 [3,X,Y,Z]`, components `(x,y,z)`, in fixed-grid voxel units,
  with the sampling convention `moving(fixed_coordinate + displacement)`.
- Save a NIfTI DVF with the fixed image affine and reload it to verify shape,
  dtype, and affine before evaluation.
- COPD's 300 paired landmarks are evaluation-only. R231 whole-lung masks are
  shape/affine/QC-only. Neither landmarks nor R231 masks may enter the loss.

### Intensity rule

`p2e_equiv_hu` is the physical-HU representation of the historical P2e raw
stored-value window `[80,900]` when `HU = stored_value - 1000`:

```text
low_hu  = -920
high_hu = -100
normalized = clip(HU, -920, -100) transformed linearly to [0,1]
```

Do **not** apply `[80,900]` directly to physical HU. That is a different
window and caused the superseded early COPD audit to collapse most lung voxels
to zero.

### Optimizer and multiresolution settings

| Setting | Frozen value |
| --- | --- |
| Method | direct cubic B-spline control-point optimization; no INR |
| Seed | `1` for NumPy and PyTorch |
| Optimizer | Adam |
| Learning rate | `1e-3` at every stage |
| Full control spacing (`full_cps_vox`) | `64,32,16,8,4` |
| Image downsample | `16,8,4,2,1` |
| Stage epochs | `400,400,400,400,900` (2,500 total) |
| Derived control shapes `[Z,Y,X]` | `8,7,8`; `12,10,12`; `20,16,20`; `36,28,36`; `68,52,68` |
| Pyramid shapes `[Z,Y,X]` | `14,12,14`; `28,24,28`; `56,48,56`; `112,96,112`; `224,192,224` |
| Prolongation | exact crop-aligned cubic B-spline dyadic knot insertion |
| Interpolation | `grid_sample(..., align_corners=True)` |
| Anti-aliasing | sigma scale `0.5`, radius sigma `3.0` |

At each stage, expand the current controls **on that stage's pyramid grid**,
warp the moving image, evaluate the loss, backpropagate, and update the
controls. This stage-grid expansion is necessary and must remain.

### Objective

| Component | Frozen value |
| --- | --- |
| Similarity | pTV-style local CC |
| Local-CC sigma | `2.1 mm` |
| Local-CC radius | `2.0 sigma` |
| Local-CC sigma floor | `0.8` pixels |
| Loss domain | full pair-local grid, excluding only a 5-voxel image border |
| Regularizer | physical isoTV on the control representation |
| isoTV weight | `0.11` |
| isoTV `csqrt` | `0.005` |

### Post-registration artifacts and metrics

The final full-grid DVF is required exactly once to produce the following
post-registration artifacts:

- `fields/<case>_DVF.nii.gz` and a NumPy `[3,X,Y,Z]` field copy;
- stage history and per-stage summary, including actual epochs and timing;
- per-landmark COPD TRE: trilinear field sample at each fixed landmark,
  `fixed + u`, then physical-world-coordinate Euclidean error in mm;
- full-grid P2e regularity and field-magnitude metrics;
- challenge-style whole-grid folding from the unchanged Learn2Breath
  evaluator; and
- a complete `summary.json` with configuration, source/environment provenance,
  timing, output-field contract, TRE, and QA results.

## Required full-grid-expansion correction

### What remains required

1. The per-epoch displacement on the current loss grid at every stage.
2. The final-stage full-grid displacement for DVF export and all post-training
   evaluation.

### What must be removed from the baseline path

The historical pair-local runner unconditionally performs this operation after
**every** stage:

```python
final_dense = expand_on_shape(control, control_shape, full_cps, image_shape, 1)
```

For stages 0--3, that field is overwritten by the next stage. Exact
prolongation uses `prev_control` and `prev_shape`; it does not consume the
previous full-grid field. Replace the unconditional operation with an explicit
final-stage condition:

```python
is_final_stage = stage_index == len(STAGE_FULL_CPS) - 1
if is_final_stage:
    with torch.no_grad():
        final_dense = expand_on_shape(control, control_shape, full_cps, image_shape, 1)
```

Record `full_grid_dvf_materialized` in each stage summary. A corrected
baseline must report:

```text
false, false, false, false, true
```

If later visual/prolongation diagnostics need intermediate full-grid fields,
put them behind an explicit opt-in setting such as
`save_intermediate_full_grid_dvf`. Label their timing as diagnostic overhead,
save them outside the baseline timing result, and keep the setting disabled for
the all-10 retiming cohort.

## Corrected all-10 timing protocol

### Worker model

Create an immutable ten-job manifest for the frozen COPD/P2e-equivalent
configuration. Start one fresh Python/CUDA worker on one designated GPU and
run the jobs sequentially in this order:

| Sequence index | Case | Temperature |
| ---: | ---: | --- |
| 0 | 104 | cold |
| 1--9 | 105--113 | warm |

"Warm" means that Python, CUDA, and relevant library paths were initialized by
the preceding case. It does **not** mean registration warm-starting: every
case starts from zero controls and its own seed-1 optimization.

Run an independent preflight over all ten jobs before launch. Preserve the
historical full-case definition by also allowing each registration to perform
its own preflight before optimizer timing begins.

### Required timing fields

Write the following per case, with definitions embedded in the summary:

| Field | Definition |
| --- | --- |
| `optimizer_elapsed_sec` | synchronized P2e optimizer plus image-pyramid work only |
| `full_case_elapsed_sec` | load, preflight, normalization, optimizer, DVF export/reload, evaluation, regularity, and result writes; excludes queue time |
| `worker_case_elapsed_sec` | process-level time around the whole case invocation |
| `worker_process_start_to_case_complete_sec` | cumulative persistent-worker time |
| `stage_summaries[*].elapsed_sec` | stage-specific elapsed time and full-grid materialization role |

Also record GPU model/UUID, CUDA and PyTorch versions, visible device mapping,
CPU-thread environment, peak GPU memory, code hashes, input checksums or
manifest identifiers, and output paths. Aggregate cold, warm (cases 105--113),
and all-case mean/sample-SD/min/max summaries for all three timing fields.

The completed cohort is context, not an acceptance threshold: it measured
`35.97 +/- 1.92 s` optimizer time and `320.02 +/- 117.43 s` full-case time per
case. Its full-case measure includes evaluation/output work and redundant
non-final expansions; timing from another GPU or software stack is not a
direct absolute comparison.

### Smoke and acceptance checks

1. Run case 104 with a separate five-stage `1,1,1,1,1` smoke schedule; do not
   mix this output with the cohort results.
2. Assert the materialization trace is `false,false,false,false,true` and that
   exact prolongation completes without an intermediate dense field.
3. Require finite losses, controls, final DVF, TRE, magnitude, regularity, and
   challenge-folding metrics.
4. Reload every final DVF and verify `float32 [3,224,192,224]` plus the fixed
   affine.
5. Run the frozen 2,500-step configuration for all ten cases only after the
   smoke succeeds. Report per-case TRE deltas against the completed cohort as
   regression information, with environment provenance; do not conceal a
   material deviation by replacing the reference row.

## OASIS MR-D-BSCP acceleration playbook (documentation only)

### Boundary of applicability

The following strategies were tested for OASIS MR-D-BSCP with controls
`8->4->2`, downsampling `4->2->1`, 2,500 maximum updates, and the OASIS image
registration objective. They are not validated COPD P2e settings: the data,
grid, stage count, loss, regularizer, and evaluation differ. Use them only as
a clearly named future adaptation menu. Any COPD acceleration candidate must
be a separately registered protocol/screen and must never silently replace the
frozen rerun above.

Source of truth for this section:

- `scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py`
- `scripts/experiments/oasis_alpha_sweep/run_one.py`
- `experiments/2026-08-13_oasis_unified_acceleration_sweep/`

### Complete tested MR-D-BSCP matrix

| Variant | Stage LR | Stage epoch caps or fixed budgets | Adaptive parameters |
| --- | --- | --- | --- |
| `MR_baseline_lr1x_full` | `1e-3,1e-3,1e-3` | fixed `400,800,1300` | none |
| `MR_F60_lr3x` | `3e-3,3e-3,3e-3` | fixed `240,480,780` | none |
| `MR_F40_lr10x` | `1e-2,1e-2,1e-2` | fixed `160,320,520` | none |
| `MR_F25_lr30x` | `3e-2,3e-2,3e-2` | fixed `100,200,300` | none |
| `MR_A_bal_lr10x` | `1e-2,1e-2,1e-2` | caps `400,800,1300` | min `60,120,195`; patience `25,50,80`; interval `25`; total/data thresholds `5e-5,5e-5,5e-5` / `2e-5,2e-5,2e-5` |
| `MR_A_fast_lr30x` | `3e-2,3e-2,3e-2` | caps `400,800,1300` | min `40,80,130`; patience `20,35,50`; interval `25`; total/data thresholds `1e-4,1e-4,1e-4` / `5e-5,5e-5,5e-5` |
| `MR_F60_decay` | `3e-2,1e-2,3e-3` | fixed `240,480,780` | none |
| `MR_F40_decay` | `3e-2,1e-2,3e-3` | fixed `160,320,520` | none |
| `MR_F25_decay` | `3e-2,1e-2,3e-3` | fixed `100,200,300` | none |
| `MR_A_bal_decay` | `3e-2,1e-2,3e-3` | caps `400,800,1300` | min `60,120,195`; patience `25,50,80`; interval `25`; total/data thresholds `5e-5,5e-5,5e-5` / `2e-5,2e-5,2e-5` |
| `MR_A_fast_decay` | `3e-2,1e-2,3e-3` | caps `400,800,1300` | min `40,80,130`; patience `20,35,50`; interval `25`; total/data thresholds `1e-4,1e-4,1e-4` / `5e-5,5e-5,5e-5` |
| `MR_A_fast_decay_legacy` | `3e-2,1e-2,3e-3` | caps `400,800,1300` | min `75,150,250`; patience `50,75,100`; interval `25`; total/data thresholds `1e-4,1e-4,1e-4` / `5e-5,5e-5,5e-5` |

### Exact adaptive stopping rule

At an eligible stage epoch (at least its configured minimum and divisible by
the 25-epoch check interval), the OASIS implementation:

1. sets `window = 25` and uses `max(window, configured_patience)`;
2. computes the median total loss and median data loss over the newest
   25-epoch window;
3. computes the same medians over a preceding 25-epoch window ending one
   effective-patience interval earlier;
4. calculates relative improvements as `(previous_median - recent_median) /
   max(abs(previous_median), 1e-8)`; and
5. stops the current stage only if **both** total-loss and data-loss relative
   improvements are below their configured thresholds.

Record each stage's actual epoch count and structured plateau/stop reason.
This is stage-local early stopping, not a global 2,500-update budget stop.

### OASIS selection evidence

`MR_A_fast_decay` was selected after the OASIS first-10 screen. It used stage
LR `3e-2,1e-2,3e-3`, adaptive-fast stopping, and a mean 437.5 updates. It
showed a 60.9% full-runtime saving, mean Dice `0.8061` versus baseline
`0.8070`, HD95 `2.201` mm versus baseline `2.202` mm, and warm
loaded-images-to-DVF time `11.03 s` versus baseline `62.45 s`.

Those values justify documenting the strategy, not transferring its numeric
hyperparameters to COPD without a separately labelled COPD validation screen.
