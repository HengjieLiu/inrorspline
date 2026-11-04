# P75: First10 NCC9 method-family ARC figures

## Requested SITReg-only F/G presentation revision

Remove VFA and FeDDA from F/G only; retain SITReg and all three optimization
families, existing lambda points, shared x axes/typography and two-row ellipse
convention. Improve method separation with blue solid/circle SITReg,
vermillion dashed/square displacement, purple dash-dot/triangle SVF and a
larger black diamond for bounded-two (lambda1 only). A single-row four-method
legend replaces the crowded six-method legend. No metric or experiment changes.
Export a separate panels_FG_sitreg_v2 release, preserving A–E, old F/G and
the lambda1 table; include revised F–G and A–G PDFs. Twenty operating points
per figure,400 displayed case records,350 unique source case records.
Follow-up request: tighten only the means-only row's y ranges, shared across
F/G and retaining every operating point: Dice0.62–0.85, HD951.6–5.0mm,
ASD0.30–1.12mm. Bottom-row ranges remain unchanged to show full SD ellipses.
Explicitly label the top row as a tighter y scale; no omitted points or data
changes. A–E retain their original scales and pages.

## Approved F/G extension

Add two2×3 figures, using the same fonts/axis limits/ticks and top means-only,
bottom marginal sample-SD ellipses. F: three DL NCC9 sweeps plus842 prolonged
displacement, bounded-two at111, and velocity-regularized refined SVF842.
G: the same DL sweeps plus8421 prolonged displacement (not dense), bounded-two
at1111, and velocity-regularized refined SVF8421. Use the complete prespecified
five/seven-point sweeps; the bounded family is one point, not an invented curve.
Six method styles are identical in F/G; retain DL colors and assign unique
additional colors to displacement/bounded/SVF. Three-column/two-row legend.
Thirty operating points per figure,600 displayed case rows,450 unique source
case records (DL repeats in both figures). No new numerical experiments.
New panels_FG_v1 directory contains PNG/PDF/SVG, F–G and A–G PDF books,
CSV/JSON, provenance and acceptance. Append the original A–E PDF pages without
rerendering or rewriting original files. Preserve the lambda1 table too.
Validate complete cohorts, source hashes, statistics, ellipses, shared axes,
PDF page counts, and inspect both rendered figures before acceptance.

## Approved lambda-one table appendix

Export exactly11 ordered rows: SITReg,VFA,FeDDA; prolonged displacement842/8421;
bounded-one842/8421; bounded-two842/8421; refined-SVF velocity842/8421.
All stage weights1, same ten cases, five metrics as mean±sample SD; folding
six decimal places. Use accepted P75 source data and preserve its files.
No dense, unbounded-composition control,1151, displacement-penalty SVF or
diagnostic reintegration substitutions. Export Markdown/CSV/JSON and110
case rows into a new lambda1_table_v1 subdirectory; no GPU/registration work.

## Approved scope

Aggregation only: no registration, inference, metric reevaluation, source-result
replacement, or GPU work. Use the same seed3 shuffled ranks0–9 and canonical
Dice, HD95, pooled-area symmetric ASD, fixed-raw-positive foreground SD(log J)
over positive Jacobians, and folding percentage. Historical data are read-only.

Five independent 20×11-inch 2×3 figures: columns Dice/HD95/symmetric ASD;
mean lines/markers on top, the same means plus marginal ±1 sample-SD ellipses
below (ddof1, axis-aligned, not confidence regions).

- A: SITReg, VFA, original FeDDA-1MF NCC9; lambda0.1,1/3,1,3,10.
- B: pyramid prolonged-displacement842,8421 and842-to-dense; lambda
  0.01,0.1,0.3,1,3,10,100 applied to every stage.
- C/D: matched P70 unbounded-one, bounded-one and bounded-two retained-map
  composition at842/8421, respectively; all active lambda1. These are three
  fixed-setting markers per figure, not sweep curves. Do not substitute the
  prolonged unbounded spline, P69 rerun,1151 vector, or bounded-three method.
- E: P74 refined SVFs842/8421, velocity/displacement penalties, all seven
  global lambdas; primary seven-step integration only, continuation_v2 release.

The exact scope is70 operating points/700 case records. DL lambda1 receives
the same visual styling as other points; historical provenance is retained
in data and README, not a distinct legend entry or marker.

## Shared styling

All30 axes share x limits0–2, ticks every0.4. Dice limits0.5–0.9/ticks0.1;
HD95 limits1–7/ticks1; ASD limits0–1.6/ticks0.4. Both rows and all figures use
the same labels/ticks/ranges. No clipping, smoothing or independent autoscale.
Titles24pt, column titles21pt, labels19pt, ticks/legend16pt, lines2.5pt,
markers7pt. One external figure-level legend; low-opacity SD ellipses.

## Validation and deliverables

Standalone report script and tests; validate source acceptance, available
report hashes, CSV/JSON agreement, case IDs/direction, finite metrics, exact
configuration coverage, sample SD, source summary parity, shared axes,
top/bottom mean equality, ellipse dimensions/counts, and rendered text bounds.
Record input/output hashes, source revision/dirty state, environment versions,
and unchanged source-file hashes after rendering. Inspect all five PNGs.
Export PNG/PDF/SVG, a five-page PDF, per-case and summary CSV/JSON, plot-axis
metadata, README, and checksummed acceptance under
`ARTIFACT_ROOT/repo_experiments/oasis_seed3_first10_ncc9_method_family_arc_panels_v1`.

## Limitations

Exploratory first10 tuning, not an unbiased generalization estimate. Historical
DLlambda1 is not a fresh matched-seed training. Positive-J spread excludes
folds; keep folding in accompanying data. Two-map settings receive more
optimizer budget. Equal nominal lambda across representations/penalties is
not equal-strength regularization. Shared broad axes intentionally compress
C/D and the high-accuracy region. No timings are compared in these figures.
