# OASIS Seed-3 Next10 Five-Model Alpha Sweep Technical Log

## Status

- Full next10 launch prepared on 2026-06-26 on shenggpu9.
- Full launch started 2026-06-26 00:33 Pacific via `nohup docker exec`.
- Full launch completed 2026-06-26 04:16 Pacific with `300/300` summaries,
  `300/300` `DONE` markers, and aggregation PASS.
- Launch container: `inrorspline_oasis_seed3_next10_run_20260626`.
- Docker image: `hengjieliu96/dl_dir:py310sitreg_v1_251008`.
- Preflight passed: syntax checks, dry-run manifest, CUDA visibility, OASIS data
  checks, disk check, and 30-job rank-10 smoke.
- Outer launch log:
  `/data/hengjie/inrorspline/runs/oasis_seed3_next10_alpha_sweep_be_2500e_5models_v1/outer_logs/nohup_launch_20260626_003330.log`.

## Known Limitations Before Launch

- This is a BE-trained sweep only. Diffusion is reported posthoc as native
  analytic diffusion and forward-difference diffusion; diffusion-regularized
  training sweeps are not included.
- IDIR alpha values are not directly comparable to SINR/DirectCP alpha values
  because original IDIR BE keeps its public scaling convention.
- The next10 subset is still small and should be interpreted together with the
  completed first10 sweep before making broader OASIS claims.
- DirectCP cps2 is included as a first-class method here, unlike the first10
  run where it was completed as a later overlay.
- `loss_history_all.csv` is expected to be large because every epoch is
  retained for 300 jobs.

## Acceptance Checks Planned

- Exactly `300` summaries: 10 case ranks times 5 methods times 6 alphas.
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

## Preflight Results

- Dry-run manifest: `300` jobs, `60` jobs per method, and `30` jobs per rank
  for ranks `10-19`.
- Smoke output root:
  `/database/inrorspline/runs/smoke_oasis_seed3_next10_alpha_sweep_20260626_002501`.
- Smoke review root:
  `experiments/2026-06-26_oasis_seed3_next10_alpha_sweep/smoke_20260626_002501/`.
- Smoke result: `30/30` summaries, `30` population rows, `errors=[]`.

## Completion Results

- Launch summary: `job_count=300`, `failures=[]`, elapsed `13343.8` seconds.
- Review report generated 2026-06-26 04:16 Pacific.
- Aggregation acceptance: PASS.
- Completed artifacts include `summary_runs.csv/json`, `population_summary.csv/json`,
  `population_label_metrics.csv`, per-case metrics, regularity audits,
  BE/diffusion audits, population ARC/contact-sheet PDFs, and loss curves.
