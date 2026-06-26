# Fit-Modes Paper Figure IDIR Eval Curves Execution Spec

## Objective

Add the missing IDIR matched image-similarity curves for the paper figure
without changing accepted training objectives or overwriting existing OASIS and
LungCT fit-mode results.

## Runs

- OASIS rank `9`, original sorted index `88`, fixed/moving `(0173,0136)`:
  IDIR original, alpha `1`, lr `1e-4`, 2500 epochs, GPU `0`.
- LungCT case `01`: IDIR pTV-preprocessed fixed-T00 lung-mask sampling,
  `alpha_bending=10`, lr `1e-4`, 2500 epochs, GPU `1`.
- LungCT case `08`: same LungCT IDIR protocol, `alpha_bending=10`, lr `1e-4`,
  2500 epochs, GPU `2`.

## Matched Similarity Eval

- OASIS writes `image_similarity_eval.csv` with full-image global NCC at epochs
  `1,10,20,...,2500`.
- LungCT writes `image_similarity_eval.csv` with pTV LCC data loss at epochs
  `1,10,20,...,2500`, using the same pTV LCC fixed cache and 5-voxel border
  mask convention as the SINR/DirectCP pTV LCC runs.
- Eval rows are posthoc diagnostics only: `affects_training=false`.
- IDIR training remains unchanged: sampled NCC plus IDIR bending energy.

## Paths

- Output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1`
- Review root:
  `experiments/2026-06-26_fit_modes_paper_figure/`
- Existing source folders for later notebook:
  - `experiments/2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep/`
  - `experiments/2026-06-26_lungct_case01_ptvreg_fit_modes_lr_sweep/`
  - `experiments/2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep/`

## Launch Commands

Run inside Docker container `py310sitreg_v1_test1` from
`/homebase/code_sync/inrorspline`. Launch the three commands concurrently.
Use `python -u` so milestone eval rows are visible in logs during monitoring.

```bash
python -u scripts/experiments/oasis_alpha_sweep/run_one.py \
  --method idir_original \
  --alpha 1 \
  --lr 1e-4 \
  --gpu 0 \
  --case-rank 9 \
  --epochs 2500 \
  --eval-interval 250 \
  --image-sim-eval-interval 10 \
  --variant fit_modes_paper_curves_idir_eval_v1 \
  --out-root /database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1/oasis_rank009 \
  --expected-original-case-index 88 \
  --expected-fixed-id 0173 \
  --expected-moving-id 0136 \
  --run-name oasis_rank009_idir_a1_lr1e-4_global_ncc_eval
```

```bash
python -u scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method idir_be_default_lungmask \
  --case-id 1 \
  --gpu 1 \
  --epochs 2500 \
  --lr 1e-4 \
  --alpha-bending 10 \
  --eval-interval 250 \
  --image-sim-eval-interval 10 \
  --out-root /database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1/lungct_case01 \
  --run-name lungct_case01_idir_lr1e-4_ptv_lcc_eval
```

```bash
python -u scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method idir_be_default_lungmask \
  --case-id 8 \
  --gpu 2 \
  --epochs 2500 \
  --lr 1e-4 \
  --alpha-bending 10 \
  --eval-interval 250 \
  --image-sim-eval-interval 10 \
  --out-root /database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1/lungct_case08 \
  --run-name lungct_case08_idir_lr1e-4_ptv_lcc_eval
```

## Acceptance

- Compile modified OASIS and LungCT runners inside Docker.
- Run a 20-epoch smoke with `--image-sim-eval-interval 10` and separate smoke
  output root.
- Full run produces 3 completed summaries and 3 `image_similarity_eval.csv`
  files.
- Each eval CSV has rows for epoch `1`, every tenth epoch, and epoch `2500`.
- Existing accepted fit-mode roots are not overwritten.
