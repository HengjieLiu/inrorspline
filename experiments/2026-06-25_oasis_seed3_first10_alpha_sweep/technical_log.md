# OASIS Seed-3 First10 Alpha Sweep Technical Log

## Status

- Full first10 launch started 2026-06-25 07:54 Pacific on shenggpu9.
- Launch container: `inrorspline_oasis_seed3_first10_run_20260625`.
- Docker image: `hengjieliu96/dl_dir:py310sitreg_v1_251008`.
- Preflight passed: dry-run manifest, 24-job smoke, CUDA visibility, OASIS data
  checks, and output disk check.

## Known Limitations Before Launch

- This is a BE-trained sweep only. Diffusion is reported posthoc as native
  analytic diffusion and forward-difference diffusion; diffusion-regularized
  training sweeps are not included in this run.
- DirectCP cps4 uses only lr `1e-3`. If DirectCP is unstable or weak, LR remains
  a possible confound because lr `1e-2` is not included in this first main run.
- IDIR alpha values are not directly comparable to SINR/DirectCP alpha values
  because original IDIR BE keeps its public scaling convention.
- The first10 subset is small. Population ARC curves are appropriate for the
  project first target, but final claims should still be phrased as first10
  evidence unless a larger OASIS subset is run.
- `loss_history_all.csv` is expected to be large because every epoch is retained
  for 240 jobs.
- Dense displacement output is retained for every job, so large output storage
  should be budgeted at roughly `25-30 GB`.

## Acceptance Checks Planned

- Exactly `240` summaries: 10 case ranks times 4 methods times 6 alphas.
- No missing, duplicate, or unexpected `(case_rank, method, alpha, lr)` combos.
- Each rank resolves to the expected seed-3 shuffled pair.
- Every run has `DONE`, `summary.json`, `history.csv`, case metrics,
  regularity metrics, BE/diffusion audit CSVs, model state, mask audit, and
  dense displacement output.
- IDIR summaries must use `fixed_image_positive` training samples and
  coordinate-autograd native analytic BE/diffusion.
- SINR/DirectCP summaries must use no image-loss foreground mask and B-spline
  native analytic BE/diffusion.
- Population ARC curves must aggregate cases by method/alpha before plotting
  the Pareto/front curves.

## DirectCP cps2 Preliminary Add-On

- Requested 2026-06-25 after the completed P1 first10 run.
- Launch started 2026-06-25 11:34 Pacific via `nohup docker exec` in
  `inrorspline_oasis_seed3_first10_run_20260625` after a 2-epoch smoke job
  passed.
- Launch completed 2026-06-25 11:57 Pacific with `30/30` summaries and
  `failures=[]`; overlay report and figures completed 2026-06-25 11:58 Pacific.
- Scope: DirectCP cps2 only, alpha `100,1000,10000`, 2500 epochs, seed `1`,
  eval interval `250`, lr `1e-3`.
- Purpose: preliminary overlay against the completed P1 Pareto curves.
- Output root:
  `/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1`.
- Overlay review root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_prelim_overlay/`.
- This is a three-alpha preliminary subset, not a full six-alpha DirectCP cps2
  sweep. Original P1 report/figure files must remain unchanged; overlay outputs
  use suffixed filenames under the add-on review subdirectory.

## DirectCP cps2 Six-Alpha Completion Add-On

- Requested 2026-06-25 after reviewing the P1c preliminary overlay.
- Launch started 2026-06-25 15:18 Pacific via `nohup docker exec` in
  `inrorspline_oasis_seed3_first10_run_20260625` after a 2-epoch alpha `0`
  smoke job passed.
- Launch completed 2026-06-25 15:41 Pacific with `30/30` summaries,
  `30/30` `DONE` markers, and `failures=[]`; full overlay aggregation
  completed 2026-06-25 15:43 Pacific.
- Scope: complete DirectCP cps2 by running the missing spline-grid alphas
  `0,10,100000` for ranks `0-9`, then combine those rows with the completed
  P1c alphas `100,1000,10000`.
- Fixed settings: DirectCP cps2, cubic B-spline analytic BE/diffusion, 2500
  epochs, seed `1`, eval interval `250`, lr `1e-3`.
- Planned launch pattern: one case wave at a time, alpha `0` on GPU `1`, alpha
  `10` on GPU `2`, and alpha `100000` on GPU `3`.
- Completion output root:
  `/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_completion_a0_10_100000_v1`.
- Full overlay review root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_full_overlay/`.
- Original P1 reports and P1c preliminary overlay files must remain unchanged;
  full-overlay outputs use a separate review directory and `_with_direct_cp_cps2_full`
  filename suffixes.
- Acceptance result: combined DirectCP cps2 coverage is `60/60`; each alpha
  `0,10,100,1000,10000,100000` has 10 summaries; each rank `000-009` has 6
  DirectCP cps2 summaries; full overlay report says PASS.
