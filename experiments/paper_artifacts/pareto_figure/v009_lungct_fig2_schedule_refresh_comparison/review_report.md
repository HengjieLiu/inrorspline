# LungCT Figure 2 Schedule Refresh Plot Comparison

This review compares the previous 2400-step MR-D-BSCP rows against the
new 2500-step schedule rows in the LungCT Figure 2 TRE/stdLogJ panel.

The unchanged rows are IDIR (IDIR loss) and pTVReg. The swapped rows are
MR-D-BSCP (IDIR loss) and MR-D-BSCP (pTVReg loss).

## Outputs

- `lungct_fig2_left_bottom_schedule_refresh_2x2.png`
- `source_rows.csv`
- `manifest.json`

## Headline Deltas

| Method | Reg | Old TRE | New TRE | Delta | Old stdLogJ | New stdLogJ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MR-D-BSCP (IDIR loss) | a=1000 | 1.017 | 1.016 | -0.001 | 0.347456 | 0.345099 |
| MR-D-BSCP (IDIR loss) | a=3000 | 0.986 | 0.985 | -0.001 | 0.255500 | 0.255184 |
| MR-D-BSCP (IDIR loss) | a=10000 | 0.966 | 0.967 | +0.001 | 0.185452 | 0.185390 |
| MR-D-BSCP (IDIR loss) | a=30000 | 0.983 | 0.983 | -0.000 | 0.145680 | 0.145675 |
| MR-D-BSCP (IDIR loss) | a=100000 | 1.013 | 1.013 | -0.000 | 0.118115 | 0.118107 |
| MR-D-BSCP (pTVReg loss) | w=0.011 | 1.022 | 0.978 | -0.044 | 0.397449 | 0.392750 |
| MR-D-BSCP (pTVReg loss) | w=0.033 | 0.938 | 0.936 | -0.002 | 0.267959 | 0.267836 |
| MR-D-BSCP (pTVReg loss) | w=0.055 | 0.917 | 0.917 | -0.001 | 0.223834 | 0.223713 |
| MR-D-BSCP (pTVReg loss) | w=0.11 | 0.922 | 0.924 | +0.002 | 0.179961 | 0.179973 |
| MR-D-BSCP (pTVReg loss) | w=0.33 | 0.952 | 0.952 | -0.000 | 0.133214 | 0.133221 |

## Notes

- Both columns use identical x/y limits.
- Ellipse full width/height is `2 * sample SD`, matching the current
  LungCT panel convention.
- Top row has no point annotations. Bottom row labels hyperparameter
  ratio relative to the best TRE point within each displayed curve.
- This artifact does not update Figure 2 plotting source.
