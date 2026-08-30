# v004 Diagnostic Comparison

v004 starts from v003 and changes only DIR-LAB Panel A single-stage BSCP operating points. The change makes the DIR-LAB registration diagnostics regularity-matched by foreground `SD(log J)` rather than using the same nominal FD-BE weight for both single-stage methods.

| Method | v003 alpha | v004 alpha | Mean Pareto TRE (mm) | Mean label TRE (mm) | Mean foreground SDlogJ |
|---|---:|---:|---:|---:|---:|
| INR-BSCP cps4 | 10000 | 20000 | 1.474 | 1.712 | 0.180 |
| D-BSCP cps4 | 10000 | 50000 | 1.306 | 1.547 | 0.159 |

`Mean label TRE` is computed from `tre_snap_to_voxel.tre_mean_mm`, the same metric printed on the DIR-LAB Panel A figures. `Mean Pareto TRE` is retained from the alpha-landscape source rows because that is the value paired with foreground `SD(log J)` in the Pareto selection table.

All other records are unchanged from v003. v003 remains the fixed-domain but not DIR-LAB-regularity-matched comparison artifact.
