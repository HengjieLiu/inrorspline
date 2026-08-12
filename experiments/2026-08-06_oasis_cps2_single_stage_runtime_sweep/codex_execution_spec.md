---
doc_type: codex_execution_spec
status: active
---

# OASIS cps2 Single-Stage Runtime Sweep Execution Spec

## Objective

Screen faster first10 OASIS variants for:

- `direct_cp_cps2_be_bspline_analytic` (`D-BSCP cps2`)
- `sinr_cps2_be_bspline_analytic` (`SINR cps2`)

The screening run uses full evaluation/output mode so Dice, HD95, ASD, and
regularity metrics are available for candidate selection. Sequential
persistent-process DVF-only timing is run only after promising candidates are
identified.

## Protocol

- Dataset: OASIS BrainMR seed-3 shuffled `pair_test_200`, ranks `0-9`.
- Pair file: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Foreground rule: fixed raw image-positive foreground, `fixed_image > 0`.
- Alpha: `1000`.
- Seed: `1`.
- Evaluation interval: `250`.
- Screening output root:
  `<container-data-root>/inrorspline/runs/oasis_cps2_single_stage_runtime_sweep_alpha1000_v1`.
- Persistent DVF timing root:
  `<container-data-root>/inrorspline/runs/oasis_cps2_single_stage_persistent_dvf_timing_alpha1000_v1`.
- Review root:
  `experiments/2026-08-06_oasis_cps2_single_stage_runtime_sweep/`.

## Candidate Gate

A non-baseline candidate is promising if:

- same-method first10 Dice drop is `<= 0.02` versus the rerun cps2 baseline;
- same-method HD95 increase is `<= 0.2 mm`;
- and either full-pipeline total runtime improves by at least `25%`, or epochs
  are reduced by at least `25%` with no Dice/HD95 loss.

Existing cps4 and MR-D-BSCP rows are contextual comparisons only; cps2
candidates are selected against same-method cps2 baselines.

## Commands

Compile:

```bash
python -m py_compile \
  scripts/experiments/oasis_alpha_sweep/run_one.py \
  scripts/experiments/oasis_alpha_sweep/launch_oasis_cps2_runtime_sweep.py \
  scripts/experiments/oasis_alpha_sweep/aggregate_oasis_cps2_runtime_sweep.py \
  scripts/experiments/oasis_alpha_sweep/run_oasis_cps2_persistent_dvf_timing.py \
  scripts/experiments/oasis_alpha_sweep/analyze_oasis_cps2_persistent_dvf_timing.py
```

Dry run:

```bash
python scripts/experiments/oasis_alpha_sweep/launch_oasis_cps2_runtime_sweep.py \
  --dry-run
```

Screening smoke:

```bash
python scripts/experiments/oasis_alpha_sweep/launch_oasis_cps2_runtime_sweep.py \
  --case-ranks 0 \
  --epoch-override 1 \
  --out-root <container-data-root>/inrorspline/runs/oasis_cps2_single_stage_runtime_sweep_alpha1000_v1_smoke \
  --review-root experiments/2026-08-06_oasis_cps2_single_stage_runtime_sweep/smoke \
  --overwrite
```

Full screening:

```bash
python scripts/experiments/oasis_alpha_sweep/launch_oasis_cps2_runtime_sweep.py \
  --gpus 0,1,2,3 \
  --case-ranks 0-9 \
  --max-concurrent 4
```

Persistent DVF timing:

```bash
python scripts/experiments/oasis_alpha_sweep/run_oasis_cps2_persistent_dvf_timing.py \
  --phase smoke \
  --ranks 0,1 \
  --gpu 0 \
  --overwrite

python scripts/experiments/oasis_alpha_sweep/run_oasis_cps2_persistent_dvf_timing.py \
  --phase first10 \
  --ranks 0-9 \
  --gpu 0
```
