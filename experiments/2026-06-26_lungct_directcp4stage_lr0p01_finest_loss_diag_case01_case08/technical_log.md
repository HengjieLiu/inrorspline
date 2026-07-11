# LungCT DirectCP 4-Stage Finest-Domain Loss Diagnostic Technical Log

## Known Limitations

- This diagnostic reruns only cases `01` and `08` and only DirectCP 4-stage at
  `lr=0.01`.
- The finest-domain trace is diagnostic-only. Coarse stages still train on their
  downsampled stage-domain objective.
- The diagnostic evaluates pTV LCC + isoTV on the full 1mm crop with the same
  5-voxel border-mask convention as the pTV-style training loss.

## Protocol

- Branch: `main`.
- Git commit at planning time: `b641a50`.
- Docker container: `py310sitreg_v1_test1`.
- Docker image: `hengjieliu96/dl_dir:py310sitreg_v1`.
- Data root:
  `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`.
- Output root:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1`.
- Method: `direct_cp_32_16_8_4_ptv_lcc_isotv`.
- Cases: `01`, `08`.
- Learning rate: `0.01`.
- Stage schedule: `32,16,8,4`; downsample `8,4,2,1`; epochs
  `400,400,800,900`.
- Diagnostic columns: `global_epoch`, `finest_loss`, `finest_data_loss`,
  `finest_reg_loss`, `finest_reg_weight`, `finest_loss_domain`.

## 2026-06-26 Planned Checks

- Compile the modified runner and new plotting script.
- Run two smoke jobs with `stage-epochs=1,1,1,1`.
- Verify smoke histories have finite diagnostic loss columns.
- Run full case01 and case08 jobs in parallel on GPUs `0` and `1`.
- Generate and visually inspect companion plots next to the accepted Case01 and
  Case08 sweep plots.

## 2026-06-26 Smoke Validation

- Compile passed for `run_one.py` and
  `plot_finest_domain_loss_diagnostic.py`.
- Smoke output root:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_smoke_v1`.
- Smoke settings: cases `01` and `08`, `stage-epochs=1,1,1,1`,
  `lr=0.01`, `stage-lr=0.01`, `--finest-loss-diagnostic`,
  `--finest-loss-every 1`.
- Smoke validation passed: each case produced `4` `stage_history.csv` rows,
  stage-0 `downsample=8`, `finest_loss_domain=full_1mm_crop`, and finite
  `finest_loss`, `finest_data_loss`, and `finest_reg_loss`.
- Smoke plotter wrote temporary diagnostic plots under:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_smoke_plots`.
- Smoke runtime: case01 `167.5` sec, case08 `238.0` sec. The diagnostic is
  substantially slower than the original training-only runner because coarse
  stages evaluate full-grid pTV LCC.

## 2026-06-26 Full Run

- Full output root:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1`.
- Case01 run:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1/direct_cp_32_16_8_4_ptv_lcc_isotv/image_reg_direct_cp_4stage_case01_finestdiag_sched32-16-8-4_e400-400-800-900_lr0p01`.
- Case08 run:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1/direct_cp_32_16_8_4_ptv_lcc_isotv/image_reg_direct_cp_4stage_case08_finestdiag_sched32-16-8-4_e400-400-800-900_lr0p01`.
- Case01 completed `2500` rows, final `global_epoch=2500`, final
  `finest_loss=2776212.5`, elapsed `258.6` sec.
- Case08 completed `2500` rows, final `global_epoch=2500`, final
  `finest_loss=4922483.5`, elapsed `311.7` sec.
- Validation passed: all rows have finite `finest_loss`, `finest_data_loss`,
  and `finest_reg_loss`; stage-0 rows retain `downsample=8` while
  `finest_loss_domain=full_1mm_crop`.
- Companion plots written:
  - `experiments/2026-06-26_lungct_case01_ptvreg_fit_modes_lr_sweep/plots/best_lr_registration_loss_spline_methods_finest_domain_directcp4stage_lr0p01.png`
  - `experiments/2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep/plots/best_lr_registration_loss_spline_methods_finest_domain_directcp4stage_lr0p01.png`
- Visual inspection passed: both figures render continuous DirectCP 4-stage
  curves over epochs `1-2500`; original
  `best_lr_registration_loss_spline_methods.png` files were not overwritten.
