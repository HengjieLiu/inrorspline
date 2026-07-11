# Codex Execution Spec

Experiment: P5 Overnight First-Three-Case Update-Alignment Diagnostic

Date planned: 2026-06-29

## Scope

Run the first three diagnostic cases only:

- OASIS BrainMR seed-3 shuffled rank 9, fixed/moving `(0173,0136)`.
- LungCT DIR-Lab 4DCT case 01.
- LungCT DIR-Lab 4DCT case 08.

## Source Plans

- `docs/codex_prompts/260629_3_from_260629_2_overnight_update_alignment_run_plan.md`
- `docs/codex_prompts/260629_4_from_260629_1_optional_sinr_cp_kernel_followup_plan.md`

## Environment

- Docker container: `py310sitreg_v1_test1`
- Docker image: `hengjieliu96/dl_dir:py310sitreg_v1`
- Repo path in container: `/homebase/code_sync/inrorspline`
- Run output root in container:
  `/database/inrorspline/runs/update_alignment_overnight_20260629_v1`
- Review root in repo:
  `experiments/2026-06-29_update_alignment_overnight`
- GPUs: `0,1,2,3`

## Commands

Compile:

```bash
docker exec py310sitreg_v1_test1 \
  python -m compileall \
  /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/run_one.py \
  /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/run_one.py \
  /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/launch_update_alignment_overnight.py
```

Dry run:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/launch_update_alignment_overnight.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --dry-run \
  --out-root /database/inrorspline/runs/update_alignment_overnight_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_update_alignment_overnight
```

Smoke:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/launch_update_alignment_overnight.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --smoke \
  --resume \
  --out-root /database/inrorspline/runs/update_alignment_overnight_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_update_alignment_overnight
```

Full small plan with follow-up gate:

```bash
docker exec py310sitreg_v1_test1 \
  python /homebase/code_sync/inrorspline/scripts/experiments/update_alignment/launch_update_alignment_overnight.py \
  --scope first3 \
  --gpus 0,1,2,3 \
  --run-full \
  --resume \
  --auto-followup \
  --out-root /database/inrorspline/runs/update_alignment_overnight_20260629_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-29_update_alignment_overnight
```

## Acceptance

- Compile passes in Docker.
- Dry-run queues exactly 11 jobs.
- Smoke runs all 11 tiny jobs and aggregates finite alignment rows.
- Full run completes all 11 jobs.
- Aggregation writes `checkpoint_alignment_rows.csv`,
  `case_summary_rows.csv`, `sinr_directcp_similarity_rows.csv`, plots, and
  `update_alignment_report.md`.
- If promising gate passes, follow-up start status is recorded under the run
  root.
