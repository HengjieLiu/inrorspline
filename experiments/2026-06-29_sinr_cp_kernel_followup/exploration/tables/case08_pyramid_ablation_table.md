| condition | best_variant | metric_convention | tre_mean_mm | interpretation |
| --- | --- | --- | --- | --- |
| prior full multistage reference | direct_cp_4stage prior LR sweep | native/crop snap TRE | 1.0020912408341311 | Known successful full schedule reference; not metric-identical to reg1mm rows. |
| no image pyramid | ptv_nopyr_ds1-1-1-1_lr0p01 | reg1mm snap TRE | 6.478453000741572 | Control staging alone remains poor on case08. |
| partial pyramid 4,2,1,1 | fdbe_partial_ds4-2-1-1 | reg1mm snap TRE | 11.936559391130391 | Partial coarse-to-fine does not recover the hard case. |
| standard pyramid 8,4,2,1 | ptv_standard_short_lr0p01 | reg1mm snap TRE | 2.2704644993422045 | Full standard pyramid plus LR probe gives large recovery. |
