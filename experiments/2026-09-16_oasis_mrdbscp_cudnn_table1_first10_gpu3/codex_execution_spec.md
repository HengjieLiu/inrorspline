---
doc_type: codex_execution_spec
status: active
---

# GPU-3 MR-D-BSCP Table-1 cuDNN Timing Rerun

## Purpose

Rerun the paper-facing `MR_A_fast_decay` MR-D-BSCP (8-4-2) schedule on the
OASIS seed-3 shuffled first10 protocol under a paired three-arm cuDNN backend
comparison. The run preserves the historical four-arm/first5/GPU2 controller
defaults while using explicit first10/GPU3 arguments for this experiment.

## Frozen protocol

- Candidate: `MR_A_fast_decay`.
- Method: `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- Alpha: `1000`.
- Seed: `1`.
- Split: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case ranks: `0-9`, in listed seed-3 shuffled order.
- Physical GPU: `3`; logical device inside the masked process is `cuda:0`.
- Container: `py310sitreg_v1_artifact_store`.
- Image: `hengjieliu96/dl_dir:py310sitreg_v1`.
- Timing detail: `deep`.
- CPU thread limit: `6`.

The exact MR schedule is control points `8,4,2`, image downsampling `4,2,1`,
maximum stage epochs `400,800,1300`, stage learning rates `3e-2,1e-2,3e-3`,
and adaptive early stopping from the existing `MR_A_fast_decay` candidate
definition.

## Configuration order

Configurations run strictly sequentially in this order:

1. `benchmark_false_deterministic_true`
2. `benchmark_false_deterministic_false`
3. `benchmark_true_deterministic_false`

Each configuration uses one fresh worker. Within the worker, rank `0` is the
cold run, followed by rank `0` again and ranks `1-9` as ten warm runs. The
repeated rank-0 output receives a unique warm suffix.

## Timing and validation

Timing is `case_load_complete` through `dvf_file_ready`, equivalent to loaded
input images through saved dense DVF. It includes optimization and DVF output,
and excludes metric evaluation.

The full run expects `3 * 11 = 33` timing rows. It records endpoint timing,
warm mean and sample SD, cold-minus-warm gap, setup-device time, phase
breakdowns, actual steps, peak memory, requested/effective backend metadata,
and environment provenance.

After timing succeeds, the evaluator validates one cold/output DVF per unique
case and configuration: `3 * 10 = 30` Dice rows. Evaluation uses one fixed
`benchmark=False, deterministic=True` backend and is excluded from timing.

## Artifact locations

- Run root: `/database/inrorspline/runs/oasis_mrdbscp_cudnn_table1_first10_gpu3_20260916_v1`.
- Review root: `/database/inrorspline/repo_experiments/2026-09-16_oasis_mrdbscp_cudnn_table1_first10_gpu3`.
- Timing report: `mrdbscp_cudnn_ablation_report.md` under the review root.
- Timing rows: `timing_case_rows.csv` and `timing_summary.csv` under the review root.
- Dice rows: `dice_case_metrics.csv` and `dice_summary.csv` under the review root.

## Commands

All commands are executed inside the default artifact-store container from its
current mounted repository path `/homebase/code_sync/inrorspline`, with
`ARTIFACT_ROOT=/database/inrorspline` and the OASIS root mounted at
`/mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1`.

Dry run:

    python /homebase/code_sync/inrorspline/scripts/experiments/oasis_alpha_sweep/run_mrdbscp_cudnn_ablation.py \
      --gpu 3 \
      --container-name py310sitreg_v1_artifact_store \
      --oasis-root /mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1 \
      --case-ranks 0-9 \
      --configs benchmark_false_deterministic_true,benchmark_false_deterministic_false,benchmark_true_deterministic_false \
      --repeat-cold-rank \
      --variant oasis_seed3_first10_mrdbscp_cudnn_table1_gpu3_v1 \
      --out-root /database/inrorspline/runs/oasis_mrdbscp_cudnn_table1_first10_gpu3_20260916_v1 \
      --review-root /database/inrorspline/repo_experiments/2026-09-16_oasis_mrdbscp_cudnn_table1_first10_gpu3 \
      --dry-run

Smoke uses the same arguments, with `--smoke`, separate `_smoke` artifact roots,
and `--skip-dice`; it runs one one-step cold case per configuration.

The full run uses the dry-run arguments without `--dry-run` or `--skip-dice`.
