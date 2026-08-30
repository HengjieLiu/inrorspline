# v004 DIR-LAB Regularity-Matched Panel A Validation Report

- Status: `passed`
- Source index rows: `160`
- DIR-LAB Panel A regularity-matched replacements: `20/20`
- Unchanged v003 rows: `140`
- Replacements are exact-compatible reuse from the accepted FD-BE cps4 alpha-landscape run.
- INR-BSCP cps4 uses `a=20000`; D-BSCP cps4 uses `a=50000`.
- DIR-LAB MR-D-BSCP remains `400/400/800/900`, cps `32/16/8/4`, `a=10000`.
- OASIS remains unchanged from v003 except for the shared Panel A axis scale.
- Panel A axis: `1e-2–1e0`; Panel B axis: `1e-1–1e3`.
- DIR-LAB Panel A TRE labels: `40/40`.
- TRE labels use inside-plot leader lines; INR-BSCP and D-BSCP labels are placed above the curves, while INR-Dense and MR-D-BSCP labels are placed below.
- Figure 3 camera-ready 2x4 variant: `population_overlay_plus_lungct_cases_2x4_tre_labels_camera_ready_fig3.png`; uses no super-title, `DIR-LAB 4DCT` subtitles, `Step` x-axis labels, and enlarged tick/axis/legend typography.
- TRE labels use `tre_snap_to_voxel.tre_mean_mm` from each run summary; the regularity-selection comparison table reports the Pareto source TRE used with foreground SDlogJ.
