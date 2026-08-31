# COPD8 (`case_111`) P2e audit report

## Status

The corrected COPD8 audit completed on 2026-08-30 and is ready for review.
The first audit is preserved below as superseded provenance because it applied
the historical window in the wrong intensity domain. No other COPD pair or
Learn2Breath validation case was run.

## Superseding preprocessing correction

The pair-local manifest declares `HU = stored_value - 1000`. Historical P2e
applied `[80,900]` to `stored_value`, not to physical HU. The first port
mistakenly used `clip(HU, 80, 900)` and collapsed 90.6% of fixed voxels to
zero. The corrected port uses:

```text
stored = HU + 1000
normalised = (clip(stored, 80, 900) - 80) / 820
```

This preserves the selected historical P2e preprocessing in its original
value domain. The earlier field and metrics remain external diagnostic
artifacts; they are not the valid P2e-port result.

## Preflight

- Pair geometry: shared `224 x 192 x 224` grid, 1.5 mm isotropic spacing, and
  matching fixed/moving affine.
- Direction: phase 2 EXP moving to phase 1 INSP fixed.
- Landmarks: 300 ordered IDs, all in bounds. Identity physical TRE was
  **26.4270 mm** (median 23.3687 mm; p95 51.1471 mm).
- R231: fixed and moving mask geometry matched the CT grids; used for QC only.
- Field adapter: zero-flow and `[1.25, -0.75, 0.5]` XYZ-voxel known-shift
  checks passed. The NIfTI field reloaded as fixed-grid `float32 [3,X,Y,Z]`.
- The originally reported window occupancies below belong to the invalid
  direct-HU implementation and are retained only to explain its failure.

## Superseded registration result: direct-HU window

| Metric | Result |
| --- | ---: |
| Landmark evaluation | Physical-mm, trilinear field sample at each fixed landmark |
| Registered mean TRE | **27.1577 mm** |
| Registered median / p95 / max TRE | 24.9944 / 52.2465 / 60.1393 mm |
| Change from identity mean TRE | +0.7307 mm |
| Optimisation runtime | 37.96 s on GPU (posthoc full-grid regularity is separate) |
| Peak allocated GPU memory | 1.87 GiB |
| Whole-grid folding | 0.09591% |
| Whole-grid mean Jacobian / log-Jacobian SD | 0.96066 / 0.13831 |
| Mean / p95 / max displacement magnitude | 5.2484 / 10.1939 / 18.3205 voxels |

The field was finite and all required artifacts were generated, but this first
registered mean TRE is invalid as a P2e-port result because its normalisation
was wrong. It does not authorise a cohort rollout.

## Corrected registration result: historical stored-value window

The valid rerun uses the raw-value conversion above. Its preflight repeated
the pair geometry, phase direction, 300 landmark IDs/bounds, R231 geometry,
baseline TRE, and zero/known-shift field tests; all passed. It exported and
reloaded a finite fixed-grid `float32 [3,X,Y,Z]` DVF.

| Metric | Corrected result |
| --- | ---: |
| Landmark evaluation | Physical-mm, trilinear field sample at each fixed landmark |
| Baseline / registered mean TRE | **26.4270 / 1.7129 mm** |
| Registered median / p95 / max TRE | 0.9413 / 8.8858 / 19.1797 mm |
| Landmark count | 300 |
| Optimisation runtime | 41.73 s on GPU (posthoc full-grid regularity is separate) |
| Peak allocated GPU memory | 1.87 GiB |
| Whole-grid folding | 0.22037% |
| Whole-grid mean Jacobian / log-Jacobian SD | 0.92511 / 0.31501 |
| Mean / p95 / max displacement magnitude | 9.6384 / 23.0110 / 48.0762 voxels |

This is the valid single-case historical-P2e-window adaptation result. It is
not a direct numerical comparison to the historic 1-mm crop protocol.

## Visual audit

[`registration_audit.ipynb`](registration_audit.ipynb) was re-executed in the
Learn2Breath Docker environment against the corrected DVF. It has no cell
errors and contains the requested inline 2x9 coronal panel:

- CT row: moving, P2e-warped moving, fixed at fixed-R231 25%, 50%, and 75%
  anterior--posterior positions.
- R231 row: moving, nearest-neighbour P2e-warped moving, fixed at the same
  positions.

It does not save a standalone figure. The renderer exactly follows the
accepted mask-audit coronal convention: `np.fliplr(array_xyz[:, y, :].T)`,
`origin="lower"`, S/I vertical markers, and R/L horizontal markers. The
corrected external artifact root is
`${L2B_P2E_RUNTIME_ROOT}/copd_case_111_historical_raw_window/` with `fields/`,
`results/`, and `training/` subdirectories.

## Stop condition

Hold here for review. The planned ten COPD TRE runs and ten validation
Dice/folding evaluations remain unlaunched.
