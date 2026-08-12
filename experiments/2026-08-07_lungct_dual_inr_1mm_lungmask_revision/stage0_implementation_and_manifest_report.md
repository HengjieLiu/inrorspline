# Stage 0 Implementation And Manifest Report

## Status

Stage 0 implementation is complete as of 2026-08-11. The PL1C default dry-run
manifest was generated successfully, and focused compile/test checks passed in
the project Docker container.

## Implemented Changes

- Added `PL1C` to the controlled LungCT DUAL-INR variant list.
- Added loss branch identifier `ptv_lcc_isotv`.
- Kept `PL1C` on the original DUAL blur schedule `[4,2,0]` with 3000 total
  steps.
- Added pTVReg-style sampled local cross-correlation plus sampled physical-unit
  isoTV regularization for author-blur DUAL variants.
- Added explicit regularization override plumbing:
  - `regularization_multiplier` for native DUAL and bending-energy branches;
  - `ptv_isotv_weight` for PL1C.
- Added exact resolved-weight metadata to each run configuration and summary.
- Added optional `run_tag` support in `run_one.py` so later sweep points can
  avoid output-directory collisions.
- Extended focused tests for PL1C, weight scaling, and launcher variant
  matrices.
- Updated `docs/04_experiment_registry.md` with P29 before the real PL1C run.

## PL1C Default Settings

- Variant: `PL1C`.
- Protocol: `PL`.
- Image domain: pTVReg 1 mm isotropic cropped images.
- Image masking: unmasked image intensities.
- Sampling mask: fixed T00 1 mm lung mask.
- Metric mask: fixed T00 1 mm lung mask.
- Schedule: original DUAL blur sigmas `[4,2,0]`.
- Total steps: `3000`.
- Model: released two-branch DUAL SIREN.
- Loss branch: `ptv_lcc_isotv`.
- pTV local-CC sigma: `2.1 mm`.
- pTV local-CC radius: `2.0 sigma`.
- pTV local-CC sigma floor: `0.8 px`.
- Sampled LCC centers per step: `512`.
- Sampled LCC offsets per center: `64`.
- isoTV `csqrt`: `5e-3`.
- Default PL1C isoTV weight: `0.055`.
- Semantic/lobe/segmentation loss: disabled.

## Compact Sweep Matrix For Later Stages

The planned camera-ready sweep remains compact and all-case:

| Variant | Schedule | Loss | Weight points |
|---|---|---|---|
| `PL1D` | `[8,4,2,1]` image pyramid | native DUAL | multipliers `[0.01,0.1,1,10,100]` |
| `PL1B` | `[8,4,2,1]` image pyramid | IDIR-type BE | multipliers `[0.01,0.1,1,10,100]` |
| `PL1A` | `[4,2,0]` original blur | native DUAL | multipliers `[0.01,0.1,1,10,100]` |
| `PL1C` | `[4,2,0]` original blur | pTV LCC + isoTV | weights `[0.0055,0.017,0.055,0.17,0.55]` |

Each curve point must use all 10 cases. Existing default rows may be reused for
`PL1D`, `PL1B`, and `PL1A` at multiplier `1` only if the protocol, schedule,
loss, resolved weights, and posthoc regularity match.

## Reusable Existing Rows

- `PL1D`, multiplier `1`:
  `<container-data-root>/inrorspline/runs/lungct_dual_inr_1mm_lungmask_revision_v1/pl1d`.
- `PL1B`, multiplier `1`:
  `<container-data-root>/inrorspline/runs/lungct_dual_inr_1mm_lungmask_revision_v1/pl1b`.
- `PL1A`, multiplier `1`:
  `<container-data-root>/inrorspline/runs/lungct_dual_inr_1mm_lungmask_author_schedule_v1/pl1a`.

These rows are not copied into the Stage 1 PL1C run root. They should be
imported during the later curve aggregation stage if their acceptance metadata
still matches.

## Stage 1 Dry-Run Manifest

Generated manifest:

```text
experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage1_pl1c_default_resolved_job_manifest.json
```

Manifest summary:

- Stage: dry-run.
- Variants: `PL1C`.
- Cases: `01-10`.
- Planned jobs: `10`.
- GPUs: `0,1,2,3`.
- Seed: `42`.
- Output root:
  `<container-data-root>/inrorspline/runs/lungct_dual_inr_camera_ready_curve_stage1_pl1c_default_v1`.
- Review root:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/`.

## Verification

Commands run in Docker:

```bash
PYTHONPYCACHEPREFIX=/tmp/inrorspline_pycache_check python -m py_compile \
  scripts/experiments/lungct_dual_inr_controlled/adapter.py \
  scripts/experiments/lungct_dual_inr_controlled/run_one.py \
  scripts/experiments/lungct_dual_inr_controlled/launch_lungmask_revision.py \
  scripts/experiments/lungct_dual_inr_controlled/aggregate_lungmask_revision.py
```

```bash
PYTHONPATH=<repo-root> \
PYTHONPYCACHEPREFIX=/tmp/inrorspline_pycache_check \
pytest -q \
  scripts/experiments/lungct_dual_inr_controlled/tests/test_adapter.py \
  scripts/experiments/lungct_dual_inr_controlled/tests/test_lungmask_revision_launcher.py
```

Result:

```text
10 passed
```

## Known Limitations Before Stage 1

- PL1C uses the pTVReg-style sampled LCC/isoTV loss with the DUAL SIREN
  architecture; it is not a reproduced pTVReg or MR-D-BSCP implementation.
- The first PL1C full-length case is still the practical smoke test for loss
  scale and numerical stability.
- The shared historical filename `resolved_job_manifest.json` in this review
  folder was also updated by the dry-run launcher; the PL1C-specific copy above
  is the intended Stage 1 manifest reference.

## Stage 1 Decision

Proceed to PL1C case `01` at weight `0.055`. If it completes with finite TRE
and regularity, continue the same default setting on all 10 cases using all
four GPUs.
