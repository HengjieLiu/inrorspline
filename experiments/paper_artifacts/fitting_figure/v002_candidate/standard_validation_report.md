# Fitting Figure LR Validation

- Rows checked: `16`
- Blocking issues: `0`

| dataset | method_id | method_family | observed_lr_label | canonical_lr_label | status | selected_setting | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OASIS SITReg | IDIR | idir | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE in the OASIS field-fit sweep. |
| OASIS SITReg | SINR cps2 | sinr | 1e-4 | 1e-4 | matches_standard | 1e-4 | Final-control-spacing matched OASIS cps2 field-fit add-on: alpha 1000, canonical LR, masked normalized-grid MSE. |
| OASIS SITReg | DirectCP cps2 | directcp | 1e-3 | 1e-3 | matches_standard | 1e-3 | Final-control-spacing matched OASIS cps2 field-fit add-on: alpha 1000, canonical LR, masked normalized-grid MSE. |
| OASIS SITReg | DirectCP 8-4-2 | directcp_multistage | 1e-3 | 1e-3 | matches_standard | 1e-3 | Canonical OASIS multistage DirectCP field-fit add-on: cps 8/4/2, downsample 4/2/1, stage epochs 400/800/1300, lr 1e-3; plotted using finest-domain masked normalized-grid MSE diagnostics. |
| OASIS Greedy | IDIR | idir | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE in the OASIS field-fit sweep. |
| OASIS Greedy | SINR cps2 | sinr | 1e-4 | 1e-4 | matches_standard | 1e-4 | Final-control-spacing matched OASIS cps2 field-fit add-on: alpha 1000, canonical LR, masked normalized-grid MSE. |
| OASIS Greedy | DirectCP cps2 | directcp | 1e-3 | 1e-3 | matches_standard | 1e-3 | Final-control-spacing matched OASIS cps2 field-fit add-on: alpha 1000, canonical LR, masked normalized-grid MSE. |
| OASIS Greedy | DirectCP 8-4-2 | directcp_multistage | 1e-3 | 1e-3 | matches_standard | 1e-3 | Canonical OASIS multistage DirectCP field-fit add-on: cps 8/4/2, downsample 4/2/1, stage epochs 400/800/1300, lr 1e-3; plotted using finest-domain masked normalized-grid MSE diagnostics. |
| LungCT case01 | IDIR | idir | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case01 | SINR cps4 | sinr | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case01 | DirectCP cps4 | directcp | 1e-3 | 1e-3 | matches_standard | 1e-3 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case01 | DirectCP 4stage cps4 | directcp_multistage | 1e-3 | 1e-3 | matches_standard | 1e-3 | Requested LR trajectory using posthoc finest-domain masked normalized-grid MSE. Coarse stages log this diagnostic sparsely; the final 1mm stage logs every epoch. |
| LungCT case08 | IDIR | idir | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case08 | SINR cps4 | sinr | 1e-4 | 1e-4 | matches_standard | 1e-4 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case08 | DirectCP cps4 | directcp | 1e-3 | 1e-3 | matches_standard | 1e-3 | Requested LR trajectory for masked normalized-grid target-field MSE. |
| LungCT case08 | DirectCP 4stage cps4 | directcp_multistage | 1e-3 | 1e-3 | matches_standard | 1e-3 | Requested LR trajectory using posthoc finest-domain masked normalized-grid MSE. Coarse stages log this diagnostic sparsely; the final 1mm stage logs every epoch. |
