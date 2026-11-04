# P80 foreground-diffusion ARC companion

User requests a third 2×3 ARC at each depth, with foreground diffusion on x.
Reuse all accepted 100-case accuracy metrics and saved final fields. No new
optimization, inference, parameter tuning, or GPU work. Previous release stays
read-only; new source-only renderer and tests create a separate external artifact.

## Definition and limitation

Use final total displacement in voxel units for every family, not velocity or
individual increments. Forward-valid differences, squared then averaged over
components and valid edges, followed by equal averaging over three axes. Both
edge endpoints must lie within fixed raw image > 0. No crop, lambda weighting,
segmentation mask, or logarithm. Native OASIS voxels are 1mm.

This is NOT exactly the training loss: training was unmasked, and refined SVF
penalized total velocity instead of final displacement. Display this distinction
in both figures and the external readme. Also save unmasked displacement energy
and check it against the existing metric, rtol2e-6/atol1e-8.

## Scope and checks

- 4200 B-spline fields (3 families × 2 depths × 7 lambdas ×100 cases), plus
  1000 shared SITReg/VFA fields (5 lambdas each):5200 unique analyses.
- Verify accepted source checksums, case IDs, direction, field checksums,
  mask orientation and input hashes, field geometry and finite values.
- CPU-only, up to sixteen threads, per-field durable cache and failure isolation.
  Any missing/failed case prevents final acceptance/plotting; preserve failures.
- Synthetic translation/affine/boundary/unit tests, parity with training
  diffusion arithmetic, sample-SD and complete-group checks, figure tests.
- Rank0 smoke across all52 unique depth/family/lambda settings, then full cohort.
- Keep original metrics unchanged. Shared x limits/ticks across depths; y limits
  exactly match accepted SDlogJ plots. Top zoomed means; bottom ±1 sample SD
  marginal ellipses. Five consistent family styles; no special historical DL1.
- Save two figures in PNG/PDF/SVG, CSV/JSON per-case and summary, metadata,
  source/output hashes and acceptance externally under
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_first100_bspline842_8421_foreground_diffusion_arc_v1/`.

This is posthoc analysis of exploratory data, not a new registration benchmark.
