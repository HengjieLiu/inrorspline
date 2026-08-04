---
doc_type: technical_log
status: completed
experiment_id: fit_modes_population_n10_v1
---

# Technical Log

## Known limitations and warnings

- OASIS N=10 does not reproduce the reviewer's requested N=100 testing cohort.
- OASIS rank 9 was the old post hoc illustrative choice; it is retained only as
  one member of the locked first10 and for exact reuse.
- DIR-LAB registration, target fitting, and external baselines are interpretable
  together only under their documented pTVReg-equivalent crop, mask, target,
  metric, and interpolation conventions.
- Panel B measures fit to a supplied field. It is not a registration evaluation
  and does not use the Panel A regularizer despite recording alpha provenance.
- The previous `DIR-LAB` raster-label patch is a derived historical artifact;
  the notebook must be the source of the repaired figure.
- DIR-LAB MR-D-BSCP is final-spacing matched but not schedule/budget matched
  across panels: Panel A uses `200/400/800/1000` (2400 total) and Panel B uses
  `400/400/800/900` (2500 total). Panel A also has 100 fewer epochs than the
  single-stage methods. This is an open camera-ready issue, not a plotting bug.

## 2026-08-02 implementation

- Added a generalized population launcher around the audited OASIS and DIR-LAB
  one-job runners.
- Locked OASIS seed-3 shuffled ranks 0–9 and DIR-LAB cases 01–10.
- Locked the two-panel method, LR, alpha, spacing, schedule, unit, and mask
  matrix in `experiment_protocol.md`.
- Added target-power and endpoint rRMSE fields to newly generated supervised
  run summaries so aggregation does not infer them from plotted curves.
- Added an acceptance-oriented aggregator and persistent monitoring outputs.

## Run ledger

| Time (UTC) | Action | Result | Evidence |
|---|---|---|---|
| 2026-08-02 before full launch | static checks | passed | both new CLIs compiled in the declared Docker environment |
| 2026-08-02 before full launch | full dry-run and target validation | 160 unique records; OASIS 10/10 and DIR-LAB 10/10 targets passed | `job_manifest.json`, `target_validation.csv/json` |
| 2026-08-02 20:23 | 16-job smoke and resume audit | 16/16 accepted; resume recognized 16/16 as complete without relaunch | `smoke/acceptance_report.json`, `smoke/launcher_state.json` |
| 2026-08-02 20:23–22:43 | full four-GPU launch | 136 new records completed, 24 exact-compatible records reused, 0 failed; scheduler wall time 2 h 20 min 26 s | `monitor_status.md/json`, `launcher_state.json` |
| 2026-08-02 22:43; reverified 22:47 | aggregation | passed 160/160 records and 16/16 N=10 population groups with no problems | `acceptance_report.md/json`, `optimization_index.csv/json`, `population_statistics.csv` |
| 2026-08-02 | existing-figure repair | notebook titles changed from `4DCT` to `DIR-LAB`; notebook executed in Docker; PNG and manifest regenerated from source | `experiments/paper_artifacts/fitting_figure/v002_candidate/panelAB_composite/` |
| 2026-08-02 23:02 | case-level diagnostic generation | generated 20 Panel A/B figures in two 5-by-2 Markdown grids; validated all 160 plotted endpoints | `diagnostic_comparison.md`, `diagnostic_manifest.json`, `diagnostic_curve_summary.csv` |

## Decisions during execution

- Target preflight exposed a DIR-LAB root-resolution error: the audited runner
  appends `4DCT`, so the launcher must pass the parent pTVReg 1 mm root rather
  than its `4DCT` child. The launcher was corrected before smoke or full runs;
  all ten cases then passed orientation and TRE validation.
- The first failed DIR-LAB preflight remained in the compact target-validation
  files after the corrected temporary audit. After acceptance, only
  `target_validation.csv/json` were regenerated with the fixed parent root;
  the final committed audit records OASIS 10/10 and DIR-LAB 10/10 passed and
  did not alter job manifests or optimization outputs.
- The full launch required no job retry and no protocol repair. Dynamic work
  stealing kept one optimizer per GPU until the last queue wave.
- The accepted aggregate is descriptive. No paired significance/equivalence
  analysis was added post hoc.
- Thirty-seven single-stage Panel B histories ended with the loss recorded
  immediately before the last optimizer update. For those curves, the
  displayed epoch-2500 point uses the accepted post-update full-mask value from
  `summary.json`; raw and accepted terminal values are both retained in
  `diagnostic_curve_summary.csv`. Every replacement differed by less than the
  plotter's 15% fail-closed bound.
- Case-level diagnostic figures were generated only after acceptance. No final
  compact population-statistics figure or inferential analysis was added;
  those remain a separate paper-design step.
- The DIR-LAB schedule asymmetry was surfaced during visual review. The
  preferred camera-ready correction is a new, versioned Panel A MR-D-BSCP N=10
  cohort using `400/400/800/900`; the current 2400-epoch cohort must remain as
  immutable provenance. See `docs/todos/camera_ready_checklist.md`.
