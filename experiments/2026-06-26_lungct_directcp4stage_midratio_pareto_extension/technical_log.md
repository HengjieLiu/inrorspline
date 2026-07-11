---
doc_type: technical_log
status: completed
date_started: 2026-06-26
date_completed: 2026-06-26
---

# Technical Log

## Known Limitations

- This extension adds mid-ratio regularization points only. It does not rerun
  existing IDIR, pTV-LCC+isoTV, global-NCC+FD-BE, or pTVReg rows.
- The pTV-LCC+isoTV ratio anchor is fixed at `ptv_isotv_weight=0.11`.
- The global-NCC+FD-BE ratio anchor is fixed at `alpha=10000`, the best point
  from the existing all-10 FD-BE sweep.
- The IDIR ratio anchor is fixed at `alpha_bending=10`. The addendum adds
  ratios `0.3x` and `3x`, corresponding to alphas `3` and `30`.
- pTVReg remains a reference protocol point, not a clean method-only comparison.
- Foreground regularity metrics use the fixed T00 lung mask on the pTV 1mm
  crop; training losses do not use lung-mask weighting for the DirectCP rows.
- IDIR optimization uses fixed T00 lung-mask sampled NCC; the lung mask is both
  the IDIR sampling domain and the posthoc foreground regularity domain.

## Planned Runs

- pTV-LCC+isoTV:
  - weights `0.033`, `0.33`
  - cases `01-10`
  - output root `/database/inrorspline/runs/lungct_directcp4stage_midratio_ptvlcc_v1`
- global-NCC+FD-BE:
  - alphas `3000`, `30000`
  - cases `01-10`
  - output root `/database/inrorspline/runs/lungct_directcp4stage_midratio_fdbe_v1`
- IDIR addendum:
  - alphas `3`, `30`
  - run only missing alpha `3` for cases `01-10`
  - reuse existing alpha `30` summaries from
    `/database/inrorspline/runs/lungct_ptvprep_reg_sweep_idir_directcp32_v1`
  - new alpha `3` output root
    `/database/inrorspline/runs/lungct_idir_midratio_alpha3_v1`
  - use GPUs `0,1,2,3`
  - report ETA after `1/10`, `5/10`, and `10/10` new alpha `3` jobs complete

## Verification Checklist

- [x] `python -m py_compile` passes for new scripts.
- [x] pTV dry-run queues `20` jobs.
- [x] FD-BE dry-run queues `20` jobs.
- [x] pTV smoke job passes.
- [x] FD-BE smoke job passes.
- [x] pTV full batch completes `20/20`.
- [x] FD-BE full batch completes `20/20`.
- [x] Notebook executes end to end in Docker.
- [x] Final artifacts and inline figures are present.
- [x] IDIR alpha `3` dry-run queues `10` jobs.
- [x] IDIR alpha `3` smoke job passes.
- [x] IDIR alpha `3` full batch completes `10/10`.
- [x] IDIR alpha `30` reused summaries validate `10/10`.
- [x] Notebook re-executes with IDIR ratios `0.3x` and `3x`.
- [x] Final addendum artifacts validate with `230` case rows and `23` mean rows.

## Completed Preflight

- Docker compile passed for:
  - `scripts/reports/lungct_midratio_pareto_extension.py`
  - `scripts/experiments/lungct_no_sweep_repro/launch_reg_sweep_idir_directcp32.py`
  - `scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_global_ncc_fdbe_all10_sweep.py`
- pTV dry-run queued `20` jobs over GPUs `0,1,2,3`, with weights
  `0.033` and `0.33`.
- FD-BE dry-run queued `20` jobs over GPUs `0,1,2,3`, with alphas `3000`
  and `30000`.
- Smoke pTV job completed for case `01`, weight `0.033`, stage epochs
  `1,1,1,1`; summary status was `completed`.
- Smoke FD-BE job completed for case `01`, alpha `3000`, stage epochs
  `1,1,1,1`; summary status was `completed`.
- IDIR alpha `3` dry-run queued `10` jobs over GPUs `0,1,2,3`.
- Smoke IDIR alpha `3` job completed for case `01` with `epochs=1`;
  summary status was `completed`.

## Full Run Progress

- pTV-LCC+isoTV batch completed `20/20` jobs with return code 0.
- pTV-LCC+isoTV validation found weights `0.033` and `0.33`, each with cases
  `01-10`.
- FD-BE batch completed `20/20` jobs.
- FD-BE was initially launched over GPUs `0,1,2,3`. After the user requested
  GPUs `2,3` be left free, the parent launcher was stopped, already-running
  child jobs were allowed to finish, and the remaining `5` jobs were dry-run
  checked and relaunched over GPUs `0,1` only.
- FD-BE validation found alphas `3000` and `30000`, each with cases `01-10`.
- Strict aggregation produced `160` existing trained rows, `40` extension rows,
  `10` pTVReg rows, `210` combined case rows, and `21` mean rows.
- Notebook execution in Docker completed in-place:
  `pareto_visual_review_midratio_extension.ipynb`.
- Final artifact validation found 12 non-empty PNGs, no notebook execution
  errors, 12 inline PNG outputs, and pTVReg TRE validation errors below `1e-8`
  (`source_tre_validation_abs_err` max `5.55e-16`,
  `source_trestd_validation_abs_err` max `6.66e-16`).

## IDIR Addendum Completion

- IDIR alpha `3` full batch completed `10/10` jobs over GPUs `0,1,2,3`.
- ETA reports were emitted after requested checkpoints:
  - `1/10` checkpoint was reached with `4/10` complete after the first wave.
  - `5/10` checkpoint was reached with `8/10` complete after the second wave.
  - `10/10` checkpoint was reached when cases `09` and `10` finished.
- Launcher wall runtimes were approximately `300.3` sec for cases `01-04`,
  `270.3` sec for cases `05-08`, and `270.3` sec for cases `09-10`.
- IDIR alpha `3` validation found cases `01-10` complete under
  `/database/inrorspline/runs/lungct_idir_midratio_alpha3_v1`.
- Reused IDIR alpha `30` validation found cases `01-10` complete under
  `/database/inrorspline/runs/lungct_ptvprep_reg_sweep_idir_directcp32_v1`.
- Strict aggregation produced `160` existing trained rows, `60` extension rows,
  `10` pTVReg rows, `230` combined case rows, and `23` mean rows.
- Re-executed notebook validation found 12 non-empty PNGs, no notebook execution
  errors, 12 inline PNG outputs, pTVReg TRE validation errors below `1e-8`, and
  IDIR ratios `[0.01, 0.1, 0.3, 1, 3, 10, 100]`.
