---
doc_type: codex_execution_spec
status: active
---

# OASIS cps4 Single-Stage Runtime Sweep Execution Spec

## Objective

Screen faster first10 OASIS variants for:

- `direct_cp_cps4_be_bspline_analytic` (`D-BSCP cps4`)
- `sinr_cps4_be_bspline_analytic` (`SINR cps4`)

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
- Output root:
  `<external-run-root>/runs/oasis_cps4_single_stage_runtime_sweep_alpha1000_v1`.
- Review root:
  `experiments/2026-08-05_oasis_cps4_single_stage_runtime_sweep/`.

## Candidate Gate

A non-baseline candidate is promising if:

- same-method first10 Dice drop is `<= 0.02` versus the rerun cps4 baseline;
- same-method HD95 increase is `<= 0.2 mm`;
- and either full-pipeline total runtime improves by at least `25%`, or epochs
  are reduced by at least `25%` with no Dice/HD95 loss.

Existing DirectCP cps2 and SINR cps2 first10 rows are comparison anchors only;
they are not rerun in this sweep.

## Commands

Compile:

```bash
python -m py_compile \
  scripts/experiments/oasis_alpha_sweep/run_one.py \
  scripts/experiments/oasis_alpha_sweep/launch_oasis_cps4_runtime_sweep.py \
  scripts/experiments/oasis_alpha_sweep/aggregate_oasis_cps4_runtime_sweep.py
```

Dry run:

```bash
python scripts/experiments/oasis_alpha_sweep/launch_oasis_cps4_runtime_sweep.py \
  --dry-run
```

Smoke:

```bash
python scripts/experiments/oasis_alpha_sweep/launch_oasis_cps4_runtime_sweep.py \
  --case-ranks 0 \
  --epoch-override 1 \
  --out-root <external-run-root>/runs/oasis_cps4_single_stage_runtime_sweep_alpha1000_v1_smoke \
  --review-root experiments/2026-08-05_oasis_cps4_single_stage_runtime_sweep/smoke \
  --overwrite \
  --skip-aggregate
```

Full first10:

```bash
python scripts/experiments/oasis_alpha_sweep/launch_oasis_cps4_runtime_sweep.py \
  --case-ranks 0-9
```

Aggregate:

```bash
python scripts/experiments/oasis_alpha_sweep/aggregate_oasis_cps4_runtime_sweep.py
```
