# LungCT Case08 pTVReg Fit-Modes LR Sweep

Completed rows: `24/24`.
Validation: `PASS`.

## Best Rows

| Objective | Method | LR | Original Snap TRE | Masked Field MSE vox | FG logJ std | Time sec |
| --- | --- | --- | --- | --- | --- | --- |
| image_registration | IDIR | 0.0001 | 1.1195 | n/a | 0.172968 | 158.7 |
| image_registration | SINR cps4 | 0.001 | 11.7348 | n/a | 0.613257 | 176.4 |
| image_registration | DirectCP cps4 | 0.01 | 13.8108 | n/a | 0.823518 | 121.3 |
| image_registration | DirectCP 4stage cps4 | 0.01 | 1.0021 | n/a | 0.218247 | 287.9 |
| ptvreg_field_fit | IDIR | 0.0001 | 1.0394 | 0.02956031 | 0.186661 | 9.9 |
| ptvreg_field_fit | SINR cps4 | 0.0001 | 1.0433 | 0.01478699 | 0.202201 | 127.8 |
| ptvreg_field_fit | DirectCP cps4 | 0.01 | 1.0128 | 0.00123497 | 0.230608 | 84.9 |
| ptvreg_field_fit | DirectCP 4stage cps4 | 0.01 | 1.0096 | 0.00122468 | 0.230549 | 260.2 |

## Artifacts

- `summary_runs.csv` and `summary_runs.json` contain one row per completed job.
- `plots/image_registration_loss_curves.png` shows registration training losses.
- `plots/ptvreg_field_mse_curves.png` shows field-fit MSE curves.
- `plots/final_metric_vs_lr.png` summarizes final LR sensitivity.
- `plots/best_lr_masked_mse_curves.png` overlays the best-LR field-fit MSE curve for each method with log-scale y-axis.
- `plots/best_lr_registration_loss_idir.png` shows the best-LR IDIR registration loss separately because its objective scale differs.
- `plots/best_lr_registration_loss_spline_methods.png` overlays the best-LR spline registration losses, with 4-stage DirectCP plotted on a continuous 1-2500 epoch axis.

## Protocol Notes

- Field-fit MSE is optimized only inside the fixed T00 lung mask in the 1mm crop.
- Full-crop field MSE is diagnostic only.
- Original Snap TRE means native/crop-space snap-to-voxel TRE after resizing the saved 1mm displacement back to the pTVReg native landmark crop.
- The pTVReg target loader validates the source orientation by reproducing `TRE_phys=1.0142198317`.
