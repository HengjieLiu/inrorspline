# LungCT IDIR Sampled pTV-LCC + TV 0.11 Technical Log

## Status

- Planned and implemented on 2026-06-26.
- Docker target: `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Requested launch GPUs: `0,1,2,3`, even if existing jobs are present.

## Known Limitations Before Launch

- This is a pilot at the anchor weight `0.11`, not a sweep.
- IDIR has no control lattice, so the TV regularizer is a sampled analytic
  physical-unit SIREN-Jacobian analogue, not pTVReg/DirectCP control-grid
  isoTV.
- The sampled local-CC implementation is an approximation to the dense pTV-LCC
  objective. It samples local Gaussian-weighted patch offsets around pTV
  border-crop centers and scales the sampled mean to the border-crop integral.
- Training samples the pTV 1mm border-masked crop. The fixed T00 lung mask is
  used only for foreground regularity metrics.
- Existing IDIR BE, DirectCP pTV-LCC+isoTV, global-NCC+FD-BE, and pTVReg rows
  are not modified by this pilot.

## Acceptance Checks Planned

- Compile modified LungCT runner and new launcher/aggregator in Docker.
- Loader audit for cases `01-10`.
- Case01 short smoke at `ptv_isotv_weight=0.11`.
- Dry-run launcher queues exactly `10` jobs.
- Full launch completes `10/10` summaries.
- Aggregation reports no completion errors and writes primary `original_snap`
  TRE plus diagnostics.

## Launch Log

- 2026-06-26 12:39 PDT: Docker compile passed for `common.py`, `run_one.py`,
  the pilot launcher, and the pilot aggregator.
- 2026-06-26 12:39 PDT: loader audit passed for cases `01-10`;
  `all_cases_loaded=true` and `all_verification_pass=true`.
- 2026-06-26 12:40 PDT: case01 5-epoch smoke completed in
  `/database/inrorspline/runs/lungct_idir_sampled_ptvlcc_tv_w0p11_smoke_v1`.
  Losses were finite through epoch 5. Final smoke summary reported reg1mm
  snap/sub TRE `12.405/12.404` mm, elapsed `5.01` sec, and peak memory
  `0.50` GB.
- 2026-06-26 12:41 PDT: launcher dry-run passed with `requested_jobs=10`,
  `queued_jobs=10`, `skipped_completed=0`, and GPUs `0,1,2,3`.
- 2026-06-26 12:45 PDT: full launch completed `10/10` jobs with return code
  `0`. First wave cases `01-04`, second wave cases `05-08`, and final wave
  cases `09-10` all completed.
- 2026-06-26 12:45 PDT: aggregation passed with `completed_rows=10` and
  `errors=[]`. Mean original-snap TRE `1.400 +/- 0.857` mm; original-subvoxel
  TRE `1.476 +/- 0.821` mm; reg1mm snap/sub TRE
  `1.579 +/- 0.829` / `1.483 +/- 0.836` mm. Foreground stdLogJ `0.171860`,
  foreground `J <= 0` `0.0477%`, foreground BE `0.001448`, foreground
  diffusion `0.010796`, mean elapsed `31.7` sec/case, and mean peak memory
  `0.58` GB. Case08 was the outlier with original-snap TRE `3.895` mm.
