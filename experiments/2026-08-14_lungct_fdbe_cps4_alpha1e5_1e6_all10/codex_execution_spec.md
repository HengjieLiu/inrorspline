# Codex Execution Spec: LungCT FD-BE cps4 alpha extension

- Date: 2026-08-14
- Objective: run unmasked FD-BE cps4 alpha-extension lanes for INR-BSCP and D-BSCP on DIR-Lab 4DCT cases 01-10, then refresh the current TRE/SDlogJ Pareto figure.
- Output root: `/database/inrorspline/runs/lungct_fdbe_cps4_alpha1e5_1e6_all10_v1`
- Review root: `experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/`
- Figure root: `experiments/paper_artifacts/pareto_figure/v011_current_tre_stdlogj_fdbe_cps4_alpha_extension/`

## Protocol

- Dataset: canonical pTVReg iso1mm DIR-Lab 4DCT, cases 01-10.
- Direction: T50 moving image to T00 fixed image.
- Optimization loss: full-crop global NCC.
- Regularizer: dense forward-difference bending energy.
- Lung mask: fixed T00 lung mask is used for regularity and diagnostic metrics only; it does not affect optimization.
- Excluded variants: all `masked_ncc` method IDs and fixed-lung-mask loss variants.

## Launch Matrix

| GPU | Method | Implementation ID | LR | Alpha | Epochs |
|---:|---|---|---:|---:|---:|
| 0 | INR-BSCP cps4 FD-BE | `sinr_cps4_global_ncc_fd_be` | `1e-4` | `100000` | `2500` |
| 1 | INR-BSCP cps4 FD-BE | `sinr_cps4_global_ncc_fd_be` | `1e-4` | `1000000` | `2500` |
| 2 | D-BSCP cps4 FD-BE | `direct_cp_cps4_global_ncc_fd_be` | `1e-3` | `100000` | `2500` |
| 3 | D-BSCP cps4 FD-BE | `direct_cp_cps4_global_ncc_fd_be` | `1e-3` | `1000000` | `2500` |

Each GPU runs case01 through case10 sequentially for its assigned lane.

## Commands

Syntax check:

```bash
docker exec py310sitreg_v1_test1 python -m py_compile /workspace/repo/scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_extension_all10.py /workspace/repo/scripts/experiments/lungct_no_sweep_repro/aggregate_fdbe_cps4_alpha_extension_all10.py /workspace/repo/experiments/paper_artifacts/pareto_figure/build_current_tre_stdlogj_expanded.py
```

Dry-run:

```bash
docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_extension_all10.py --dry-run --gpus 0,1,2,3
```

Smoke:

```bash
docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_extension_all10.py --smoke --gpus 0,1,2,3
```

Full:

```bash
docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/launch_fdbe_cps4_alpha_extension_all10.py --resume --gpus 0,1,2,3
```

Aggregate:

```bash
docker exec py310sitreg_v1_test1 python /workspace/repo/scripts/experiments/lungct_no_sweep_repro/aggregate_fdbe_cps4_alpha_extension_all10.py
```

Refresh figure:

```bash
docker exec py310sitreg_v1_test1 python /workspace/repo/experiments/paper_artifacts/pareto_figure/build_current_tre_stdlogj_expanded.py
```
