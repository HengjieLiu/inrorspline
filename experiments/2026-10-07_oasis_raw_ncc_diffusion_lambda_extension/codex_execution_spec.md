# First10 raw-NCC/diffusion lambda extension

## Scope and interpretation

User explicitly requested additional lambda100/300/1000 on the seed3 shuffled
first10. This is a separate approved follow-up to completed P40, not a silent
expansion of its 96-fit search. All first10 cases now participate in exploratory
tuning; ranks1–9 are not held out for this follow-up. Preserve P40's artifacts.

## Fixed protocol

- Thirty fresh fits: ranks0–9 times lambda100/300/1000. Read-only reuse of each
  verified P40 camera-ready 8/4/2 prefix, plus its existing lambda30 comparison.
- Same direct absolute voxel-xyz dense parameter, pair-minmax image preprocessing,
  full-volume signed global NCC divided by the fixed initial loss (floor1e-8),
  and unmasked valid-forward-difference L2 diffusion of the total voxel field.
- Same trilinear border-padded sampling, align_corners=True. No VFA-feature loss,
  segmentation supervision, reference-field fitting, cps1 or B-spline regularizer.
- Adam LR0.3 voxels, max/min2000/300, check/window25, patience100, relative
  total/data thresholds1e-4/5e-5. Halve LR at most twice before plateau stopping.
  Evaluate lowest-total-loss checkpoint including initialization. Fresh Adam
  and seed1 for each fit; FP32, cuDNN benchmark=False/deterministic=True.
- Use existing container as UID:GID1009:1020, GPUs0–3, <=4 concurrent workers
  and <=1 new worker/GPU. Never alter existing jobs. Preserve failures; inherited
  scheduler permits one CUDA OOM retry on a released GPU with most free memory.

## Implementation and acceptance

- Reuse numerical worker/objective and scheduler; only add a backward-compatible
  worker experiment-variant setting plus a separate launcher/report generator.
- Confirm source objective/metric/prefix/scheduler hashes equal P40's hashes.
- Validate accepted source cohort, checkpoint/pair/cache/prefix/DVF hashes, IDs,
  direction, CUDA flags, dense representation and full-volume objective support.
- Unit-test exact30-job grid, only-lambda changes, unique names, disjoint paths,
  legacy CLI default, report completeness/means and selection.
- One full-volume lambda1000 rank0 Adam-step smoke through serialization and
  canonical Dice/HD95/symmetric ASD before production (not one of30 fits).
- Aggregate only after all30 finite correctly shaped DVFs have completion
  markers. Check per-run prefix hashes and unchanged prior metadata hashes.
- Report CSV/JSON/Markdown: casewise values, paired deltas versus VFA/lambda30,
  separate ranks0–9 and1–9 means, simultaneous tolerance match counts, folding,
  diffusion, stopping, GPU and runtime diagnostics. Include unchanged prefix
  and selected feature baseline. Select one lambda by first10 mean Dice, then
  ASD and HD95; separately identify each metric's best lambda. No per-case oracle.
- Artifacts use external run/review directories named
  `oasis_seed3_first10_mrdbscp842_dense_raw_ncc_diffusion_lambda_extension_v1`.

## Limitations

First10 tuning is exploratory; no unbiased validation claim. Regularization
lambda is defined against normalized NCC and voxel-unit diffusion, not earlier
normalized-coordinate bending energy. Shared-load runtimes are diagnostic.
cuDNN deterministic does not guarantee bitwise CUDA grid_sample backward.
