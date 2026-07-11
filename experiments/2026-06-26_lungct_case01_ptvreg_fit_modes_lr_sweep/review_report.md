# LungCT Case01 pTVReg Fit-Modes LR Sweep

Completed rows: `24/24`.
Validation: `PASS`.

## Best Rows

| Objective | Method | LR | Original Snap TRE | Masked Field MSE vox | FG logJ std | Time sec |
| --- | --- | --- | --- | --- | --- | --- |
| image_registration | IDIR | 0.0001 | 0.7822 | n/a | 0.097494 | 256.6 |
| image_registration | SINR cps4 | 0.001 | 0.7807 | n/a | 0.092351 | 118.3 |
| image_registration | DirectCP cps4 | 0.01 | 1.1307 | n/a | 0.243244 | 91.9 |
| image_registration | DirectCP 4stage cps4 | 0.0001 | 0.7708 | n/a | 0.100359 | 230.6 |
| ptvreg_field_fit | IDIR | 0.001 | 0.8015 | 0.02637857 | 0.115242 | 15.5 |
| ptvreg_field_fit | SINR cps4 | 0.0001 | 0.7415 | 0.00234473 | 0.098227 | 88.5 |
| ptvreg_field_fit | DirectCP cps4 | 0.01 | 0.7467 | 0.00024939 | 0.103104 | 65.7 |
| ptvreg_field_fit | DirectCP 4stage cps4 | 0.01 | 0.7467 | 0.00024918 | 0.103059 | 194.4 |

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
- The pTVReg target loader validates the source orientation by reproducing `TRE_phys=0.7671410824`.
