# LungCT 2500-Step Schedule Standardization Pilot

## Status

- pTVReg-loss validation errors: `[]`.
- Corrected IDIR-loss validation errors: `[]`.
- No full sweep has been launched by this pilot.

## pTVReg-Loss Pilot

| Row | TRE mean +/- SD | FG stdLogJ mean +/- SD | FG J<=0% |
| --- | --- | --- | --- |
| 2400-step pTVReg-loss baseline w=0.11 | 0.922 +/- 0.152 | 0.179961 +/- 0.049568 | 0.0138 +/- 0.0193 |
| 2500-step pTVReg-loss pilot w=0.11 | 0.924 +/- 0.152 | 0.179973 +/- 0.049585 | 0.0137 +/- 0.0191 |

## Corrected IDIR-Loss Pilot

| Row | TRE mean +/- SD | FG stdLogJ mean +/- SD | FG J<=0% |
| --- | --- | --- | --- |
| 2400-step corrected IDIR-loss alpha=10000 | 1.037 +/- 0.153 | 0.265395 +/- 0.063547 | 0.0349 +/- 0.0579 |
| 2400-step corrected IDIR-loss alpha=100000 | 1.003 +/- 0.154 | 0.155840 +/- 0.033335 | 0.0005 +/- 0.0014 |
| 2500-step corrected IDIR-loss alpha=33333.333333 | 1.016 +/- 0.160 | 0.195866 +/- 0.042067 | 0.0058 +/- 0.0151 |

## Decision

- pTVReg-loss pilot: schedule-stable but not better than the existing
  2400-step `w=0.11` row; useful as a standardized replacement only if
  all LungCT rows must use exactly 2500 steps.
- Corrected IDIR-loss pilot: `alpha=33333.333333` is smoother than
  `alpha=10000` but has worse TRE and less smoothness than the existing
  2400-step `alpha=100000` row; it is not the best corrected-IDIR point
  from the current evidence.
- Stop here as requested. Do not launch full 2500-step sweeps without a
  new decision.
