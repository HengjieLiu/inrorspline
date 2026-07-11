# Codex Execution Spec: SINR CP-Kernel Follow-Up

Date: 2026-06-29

Source plan:

- `docs/codex_prompts/260629_4_from_260629_1_optional_sinr_cp_kernel_followup_plan.md`
- Triggered by the full first-three-case update-alignment diagnostic if its
  automatic promising gate passes.

Execution entry point:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/run_sinr_cp_kernel_followup.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --run-full \
  --resume \
  --out-root /database/inrorspline/runs/sinr_cp_kernel_followup_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup
```

This first implementation is intentionally overnight-safe: it reuses the
deterministic sampled checkpoint fields from the update-alignment run and
computes sampled dense-update proxy metrics for SINR versus paired DirectCP.
It does not form the autograd CP kernel `J_theta J_theta^T`.

Review artifacts:

- `sinr_cp_kernel_report.md`
- `summary.json`
- `tables/kernel_metrics.csv`
- `tables/sampled_spectrum_metrics.csv`
- `plots/update_agreement_and_amplification.png`
- `plots/kernel_spectrum_optional.png`

## Exploratory Extension

After the P5 full aggregation did not pass the automatic promising gate, the
follow-up is continuing as an explicitly exploratory Scientist/Reviewer loop.

Controller:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/explore_mechanism_followup.py \
  --gpus 0,1,2,3 \
  --run-proxy \
  --launch-schedule \
  --targeted-extra \
  --stage-epochs 25,25,25,25 \
  --min-runtime-sec 14400 \
  --allow-partial \
  --out-root /database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup
```

The first smoke of no-pyramid DirectCP took roughly 4-6 minutes per job for
only `1,1,1,1` stage epochs. Therefore the exploratory launch is scoped to
LungCT cases `01` and `08` with targeted no-pyramid, partial-pyramid, and
learning-rate probes. All10 expansion is deferred until these diagnostic rows
show that the mechanism question is worth population-scale follow-up.

New expected artifacts:

- `scientist_reviewer_log.md`
- `exploration/live_monitor_status.json`
- `exploration/exploration_summary.json`
- `exploration/paper_section_draft.md`
- `exploration/tables/mechanism_triage_table.csv`
- `exploration/tables/prior_case01_case08_best_rows.csv`
- `exploration/tables/schedule_ablation_rows.csv`
- `exploration/plots/mechanism_triage_figure.png`

Sidecar monitor:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/monitor_mechanism_explore.py \
  --out-root /database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup \
  --poll-sec 900 \
  --max-runtime-sec 14400
```

The monitor does not launch, stop, or relaunch jobs. It watches the existing
manifest/status, refreshes review artifacts only after completed schedule rows
exist, and appends short Scientist/Reviewer rounds during the long run.

Second-phase standard-pyramid probe:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/run_standard_pyramid_probe.py \
  --gpus 0,1,2,3 \
  --allow-partial \
  --out-root /database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup
```

This launches case08-only DirectCP `32,16,8,4` probes with the full standard
`8,4,2,1` image pyramid after the first phase showed that no-pyramid and
partial `4,2,1,1` probes do not rescue case08 under the time-budgeted schedule.
It writes `exploration/tables/standard_pyramid_probe_rows.csv` and
`exploration/standard_pyramid_probe_report.md`.

Outcome:

- Completed `4/4`, `0` failures.
- Best standard-pyramid probe: pTV LCC + isoTV, `8,4,2,1`,
  `25,25,25,25`, LR `0.01`, case08 reg1mm snap TRE `2.270` mm.
- LR `0.001` standard-pyramid rows remained poor (`11.496` mm pTV,
  `11.361` mm FD-BE), and pTV LR `0.001` with `50,50,50,50` reached
  `8.896` mm.
- Main integrated artifacts:
  `exploration/tables/paper_ready_mechanism_table.md`,
  `exploration/tables/case08_pyramid_ablation_table.md`, and
  `exploration/plots/case08_pyramid_ablation.png`.

Refinement probe:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/run_standard_pyramid_probe.py \
  --probe-set refinement \
  --phase-name capture_refinement_probe \
  --gpus 0,1,2,3 \
  --allow-partial \
  --out-root /database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup
```

Outcome:

- Completed `4/4`, `0` failures.
- Case08 reg1mm snap TRE converged near `1.21` mm for all LR `0.01`
  e100/e200 controls: standard `8,4,2,1` e100 `1.213`, standard e200
  `1.217`, partial `4,2,1,1` e100 `1.219`, and no-pyramid e100 `1.218`.
- Main integrated artifact:
  `exploration/plots/case08_capture_refinement.png`.
