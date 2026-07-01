# Technical Log: OASIS DirectCP 8-4-2 lr=1e-3 Standardization

## Known Limitations

- This pass standardizes only the final-paper OASIS DirectCP 8-4-2
  Pareto/table LR. Existing `v001` artifacts and their temporary-exception
  records remain as historical provenance.
- The fitting Panel A/B artifacts already use canonical `lr=1e-3` for OASIS
  DirectCP 8-4-2 and are not rerun here.
- The new candidate artifacts are review artifacts until explicitly promoted.

## Execution Notes

- Planned GPU policy: use all GPUs visible inside `py310sitreg_v1_test1`, with
  one job per GPU.
- Monitoring policy: poll every 5 minutes during the full run and write
  `monitor_snapshots.jsonl` plus a rolling `monitor_latest.md`.

## 2026-07-01 Smoke

- Docker `py_compile` passed for the new launcher, aggregator, monitor,
  source-package plotter, Pareto generator, table generator, and comparison
  script.
- Visible GPUs in `py310sitreg_v1_test1`: `0,1,2,3`, all RTX 6000 Ada.
- Smoke command used rank `0`, alpha `0`, `lr=1e-3`, stage epochs `1,1,1`,
  total epochs `3`, eval interval `1`, GPU `0`, and the separate smoke root.
- Smoke result: `1/1` job completed; aggregation returned `errors=[]`; output
  includes `DONE`, `summary.json`, and `stage_history.csv` with recorded
  `lr=0.001` and `stage_epochs=[1,1,1]`.

## 2026-07-01 Full Run And Candidate Artifacts

- Full launch used GPUs `0,1,2,3` with `--max-concurrent 4`, one job per GPU.
- Monitoring snapshots were recorded every 5 minutes in
  `monitor_snapshots.jsonl`; the rolling final snapshot reports `120/120`
  completed, `0` running, `0` failed, and ETA `complete`.
- Full run result: `120/120` `DONE` markers and `120/120` `summary.json`
  files are present.
- Aggregation result: `combined_rows=120`,
  `population_scopes={"first10": 6, "first20": 6}`, and `errors=[]`.
- Source package generated at
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/pareto_first10_first20_alpha5_saved_baselines_plus_directcp842_lr1e3_paper_v1/`.
- Candidate artifacts generated at
  `experiments/paper_artifacts/best_result_table/v002_candidate/` and
  `experiments/paper_artifacts/pareto_figure/v002_candidate/`.
- Comparison guide generated at
  `experiments/paper_artifacts/comparison/v001_vs_v002_candidate/`.
- Candidate validation: best-result table has `11` `matches_standard` rows and
  `4` reference rows; Pareto has `11` `matches_standard` rows. Neither
  candidate validation contains a `temporary_exception`, `mismatch`, or
  blocking LR issue.
- Candidate paper-facing artifacts were scanned for private host/container
  paths with no hits. The operational run-review/source-package folders retain
  expected `/database` and `/homebase` provenance paths and should be treated as
  run-review artifacts, not anonymous paper-facing outputs.
- High-volume per-epoch combined history CSVs are intentionally omitted from
  the repo-side review folder by default. The per-run histories remain in the
  external run root; rerun the aggregator with `--include-history-csvs` only
  when a local deep-dive needs those combined files.
