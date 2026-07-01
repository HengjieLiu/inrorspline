# Pareto Figure v002_candidate Plot Package

- Output formats: `png`
- OASIS annotated PNGs: `81`
- OASIS paper no-annotation PNGs: `81`
- LungCT annotated PNGs: `4`
- LungCT paper no-annotation PNGs: `4`
- OASIS DirectCP 8-4-2 included: `True`
- LungCT subset: `default_exclusions`
- LungCT plotted methods: `['idir_be_default_lungmask', 'direct_cp_32_16_8_4_global_ncc_fd_be', 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'ptvreg_reference']`
- OASIS fonts: tick labels `12`, axis labels `16`, legend `12`
- LungCT fonts: tick labels `12`, axis labels `16`, legend `12`
- Blocking LR validation issues: `0`

## LungCT Exclusions

- By ratio: `{'idir_be_default_lungmask': (), 'direct_cp_32_16_8_4_ptv_lcc_isotv': (0.01, 10.0, 100.0), 'direct_cp_32_16_8_4_global_ncc_fd_be': (0.0, 0.001, 0.01)}`
- By weight: `{'idir_be_default_lungmask': (), 'direct_cp_32_16_8_4_ptv_lcc_isotv': (), 'direct_cp_32_16_8_4_global_ncc_fd_be': ()}`

## Files

- `oasis20/small_plots/`: annotated OASIS accuracy-regularity pairs.
- `oasis20/big_plots/`: annotated OASIS contact sheets and spotlight plots.
- `oasis20_paper_no_annotations/small_plots/`: OASIS pairs without alpha/reference text annotations.
- `oasis20_paper_no_annotations/big_plots/`: OASIS contact sheets and spotlight plots without alpha/reference text annotations.
- `lungct10/default_exclusions/`: annotated original snap TRE against each LungCT regularity metric.
- `lungct10_paper_no_annotations/default_exclusions/`: original snap TRE plots without ratio/pTVReg text annotations.
