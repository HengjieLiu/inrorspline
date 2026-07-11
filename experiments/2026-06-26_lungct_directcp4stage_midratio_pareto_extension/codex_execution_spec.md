# Codex Execution Spec

## Objective

Extend the LungCT all-10 DirectCP 4-stage Pareto visualization with mid-ratio
regularization points and generate a new executed review notebook without
modifying the existing visual-review notebook.

## Run Plan

- Dataset: DIR-Lab 4DCT cases `01-10`, canonical regenerated pTVReg-style
  landmark crop resampled to 1mm.
- Methods:
  - `direct_cp_32_16_8_4_ptv_lcc_isotv`
  - `direct_cp_32_16_8_4_global_ncc_fd_be`
- New pTV-LCC+isoTV weights: `0.033`, `0.33`, corresponding to ratios `0.3`
  and `3` around anchor weight `0.11`.
- New global-NCC+FD-BE alphas: `3000`, `30000`, corresponding to ratios `0.3`
  and `3` around anchor alpha `10000`.
- Schedule: `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `200,400,800,1000`, `lr=1e-3`, seed `1`.
- Launch ordering: run all pTV-LCC+isoTV jobs first; after they complete, run
  all global-NCC+FD-BE jobs.
- Parallelism: within each method batch, use GPUs `0,1,2,3` with at most four
  concurrent jobs.

## Roots

- pTV-LCC+isoTV output root:
  `/database/inrorspline/runs/lungct_directcp4stage_midratio_ptvlcc_v1`
- global-NCC+FD-BE output root:
  `/database/inrorspline/runs/lungct_directcp4stage_midratio_fdbe_v1`
- Review root:
  `experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/`
- Notebook:
  `pareto_visual_review_midratio_extension.ipynb`

## Acceptance

- Dry-runs queue exactly `20` jobs per method batch.
- Smoke jobs complete for one reduced-stage pTV-LCC+isoTV job and one
  reduced-stage global-NCC+FD-BE job.
- Full runs complete `20/20` pTV jobs before launching FD-BE, then complete
  `20/20` FD-BE jobs.
- Final notebook is executed in Docker and contains populated tables and
  `12` inline figures.
- Saved artifacts include combined case rows, combined mean rows, screening
  table CSV/Markdown, and all requested PNG figures.
