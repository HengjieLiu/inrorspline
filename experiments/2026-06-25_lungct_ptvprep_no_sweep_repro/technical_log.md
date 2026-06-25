---
doc_type: technical_log
status: completed
date_started: 2026-06-25
date_completed: 2026-06-25
---

# Technical Log

## Known Limitations

- This run uses canonical regenerated pTV-prep artifacts, not legacy task4
  in-memory crop reconstruction. Legacy task4 crop bounds were offset by one
  voxel relative to the corrected canonical pTVReg crop.
- IDIR uses the fixed T00 lung mask for sampled-coordinate training. SINR and
  DirectCP pTV methods do not use the lung mask in the loss; they use the
  pTV-style border mask. The fixed T00 lung mask is used for foreground
  regularity metrics for all methods.
- TRE is reported in two domains: original/native-crop coordinates for
  pTVReg-comparable headline TRE, and cropped 1mm registration coordinates as a
  diagnostic. Snap/subvoxel direction can flip between these domains because
  landmarks are integer in native/crop space but fractional after 1mm
  resampling.
- Single-stage SINR cps4 and DirectCP cps4 pTV LCC plus isoTV are new
  matched-loss baselines, not legacy task4 rows.

## Verification Checklist

- Loader audit: passed in `py310sitreg_v1_test1`; all 10 cases loaded, all
  canonical pTV-prep verification flags passed, fixed/moving image and mask
  shapes matched, and fixed lung masks were non-empty.
- One-epoch smoke jobs: passed for all four methods on case `01` with
  `--no-save-dense-dvf`, output root
  `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_smoke_v1`.
- Knot-insertion diagnostics for multistage run: passed. Exact-knot
  preservation errors were near numerical precision, e.g. max absolute
  preservation error `4.66e-10`, `9.31e-10`, and `1.40e-09` for the three
  stage transitions in the case `01` smoke run.
- Launcher dry-run: passed; queued exactly `40` jobs across GPUs `0,1,2,3`.
- Aggregation smoke test: passed against the four case `01` smoke summaries
  with review output under `/tmp/inrorspline_lungct_smoke_review`.

## Smoke Results

Case `01`, one training epoch or one epoch per multistage stage:

| Method | Snap TRE mm | Sub TRE mm | Elapsed sec | Peak GB |
|---|---:|---:|---:|---:|
| `idir_be_default_lungmask` | 8.070 | 8.070 | 25.6 | 4.06 |
| `sinr_cps4_ptv_lcc_isotv` | 6.586 | 6.571 | 46.2 | 2.15 |
| `direct_cp_cps4_ptv_lcc_isotv` | 3.984 | 3.916 | 42.6 | 1.33 |
| `direct_cp_32_16_8_4_ptv_lcc_isotv` | 3.652 | 3.590 | 177.4 | 3.96 |

## Full Run Results

Full launch completed `40/40` jobs with return code `0`. Aggregation produced:

```text
experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/summary_runs.csv
experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/summary_runs.json
experiments/2026-06-25_lungct_ptvprep_no_sweep_repro/review_report.md
```

Mean accuracy and fixed T00 lung-mask foreground regularity. Original TRE means
native/crop-space TRE after resizing the saved 1mm displacement back to the
corrected pTVReg native landmark crop; 1mm TRE is evaluated directly in the
cropped 1mm registration grid.

| Method | Cases | Original snap | Original sub | 1mm snap | 1mm sub | FG logJ std | FG J<=0% | BE fg | Diff fg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `idir_be_default_lungmask` | 10 | 1.017 | 1.133 | 1.236 | 1.138 | 0.142869 | 0.0000 | 0.000535 | 0.007218 |
| `sinr_cps4_ptv_lcc_isotv` | 10 | 2.904 | 3.014 | 3.143 | 3.037 | 0.301655 | 0.8918 | 0.006339 | 0.023879 |
| `direct_cp_cps4_ptv_lcc_isotv` | 10 | 6.370 | 6.450 | 6.575 | 6.495 | 0.652560 | 3.9433 | 0.034067 | 0.069535 |
| `direct_cp_32_16_8_4_ptv_lcc_isotv` | 10 | 0.922 | 1.049 | 1.168 | 1.052 | 0.179961 | 0.0138 | 0.002527 | 0.010327 |

Interpretation: IDIR and multistage DirectCP are in the expected LungCT range.
Single-stage SINR cps4 and DirectCP cps4 are weak on larger-motion cases, with
case `08` as the clearest outlier. The four-stage DirectCP run recovers stable
accuracy and low folding, but its regularity values are not intended to
byte-match legacy task4 rows because this pass reports fixed-lung-mask
foreground metrics on the canonical cropped 1mm artifacts.

## Commands

Loader audit:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/run_one.py --audit-loader
```

Launcher dry-run:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_all10_4gpu.py --dry-run --gpus 0,1,2,3
```

Full launch, after approval:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_all10_4gpu.py --gpus 0,1,2,3
```
