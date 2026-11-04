# P75 execution log

## Limitations

First10 results are exploratory. C/D are fixed-lambda points, not sweeps.
Historical DLlambda1 is visually unified but retains provenance. Positive-J
SD(log J) does not measure folding; folding accompanies all saved tables.
Composition update budgets and SVF regularization targets differ.

## Execution

- User approved five2×3 figures with shared ranges and marginal sample-SD
  ellipses only in the bottom row. No new registration or metric evaluation.
- Audited sources: P43 pyramid displacement, P70 matched composition,
  P74 continuation_v2 refined SVF, and accepted four-method DL ARC report.
- Verified expected70 operating points/700 first10 records before implementation.
- Added isolated report and independent saved-file verifier under scripts/reports;
  historical sources and reports are unchanged. Ten unit tests passed, including
  matrix rejection, sample-SD arithmetic, exact one-third aliases, correct
  compositional comparator, shared axes, uniform DL styling, fonts and ellipse
  dimensions/counts for all five figures.
- Rendered five4000×2200 PNGs plus individual SVG/PDFs and a five-page PDF.
  All30 axes passed layout bounds, top/bottom mean equality and SD extent gates.
- Visually inspected all five PNGs: readable titles/labels/legends/ticks; no
  clipping. Sparse C/D panels retain the explicitly approved shared broad axes.
  No per-point text crowding; lambda lists are given in each figure subtitle.
- Independently reconstructed all700 mean/SD values from700 CSV case records,
  verified70 groups/ten matched pairs each, CSV/JSON equality, all30 axis
  records, PNG dimensions, six SVG axes per figure, embedded PDF fonts and
  the five-page PDF. Source/report hashes remained unchanged.
- Accepted2026-10-15 20:35:00UTC. Outputs under
  ARTIFACT_ROOT/repo_experiments/oasis_seed3_first10_ncc9_method_family_arc_panels_v1.
  ACCEPTED.json records visual review and output hashes; RENDERED.json and the
  render manifest retain their original pre-review stage status for provenance.
- CPU-only execution in the existing container as1009:1020; no GPU jobs,
  registrations, inference, metric reevaluations, dependency installs or commits.

## Approved lambda-one table appendix

- Exported11 rows in the approved order: SITReg/VFA/FeDDA followed by842/8421
  pairs for prolonged displacement, bounded-one, bounded-two and refined SVF
  with velocity diffusion. No dense or unbounded-composition substitution.
- Three table-selection tests pass. An initial test expectation grouped C/C/D/D
  rather than the approved C/D/C/D schedule pairs; corrected the test string,
  not the already correct row selection, before exporting any artifacts.
- Recomputed mean/sample SD from110 matched case records; independent NumPy
  read-back verified all110 aggregate values. Source release and output hashes
  verified unchanged/consistent. Folding retains six decimal places.
- Accepted outputs: lambda1_table_v1/{table.md,summary.csv,per_case.csv,
  results.json,ACCEPTED.json} beneath the P75 artifact directory. Prior figures
  and all original report files remain byte-unchanged.

## Approved F/G extension

- Added isolated combined-figure renderer/verifier and four selection/style/
  layout tests. F/G each compare three DL NCC9 sweeps with prolonged
  displacement, bounded-two atlambda1, and velocity-regularized refined SVF
  at842/8421. Dense/displacement-penalty SVFs are explicitly excluded.
- Six distinct colors/styles, uniform DLlambda1, three-column/two-row legends,
  same20×11-inch geometry/fonts/axis limits/ticks as A–E. Bounded is a single
  high-visibility marker with its SD ellipse in the bottom row.
- Rendered/visually inspected both4000×2200 PNGs; no clipping/crowded legend.
  Produced individual PDF/SVG and a two-page F–G PDF. Existing container lacks
  PDF merger libraries/tools, so used existing host pdfunite (no installation)
  to append original A–E pages and new F–G pages into a seven-page PDF.
- Independent saved-file verifier reproduced600 mean/SD statistics from600
  displayed records (450 unique records; the same150 DL records repeat),
  verified exact original metrics,60 groups,12 axes, image/PDF/SVG structure,
  embedded fonts, source hashes and unchanged original figures/lambda1 table.
  Extracted text of the first five merged PDF pages exactly matches original.
- Accepted2026-10-15 21:06:45UTC under panels_FG_v1. No registrations,
  inference, field metric reevaluation, GPU jobs or original source edits.
- Final regression suite: all17 A–E, lambda1-table and F/G tests passed;
  git diff --check passed. No files were staged or committed.

## SITReg-only F/G revision with tighter top-row y axes

- Removed VFA/FeDDA from F/G only, directly selecting the accepted F/G saved
  data. No new registrations, inference, metric reevaluation or GPU work.
- User added a top-row y-zoom request during implementation. Shared tighter
  ranges across F/G: Dice0.62–0.85, HD951.6–5.0mm, ASD0.30–1.12mm. Every mean
  stays visible. All x axes and bottom full-SD y ranges remain unchanged;
  subtitle and row label explicitly disclose the differing row scales.
- Blue solid/circle SITReg, vermillion dashed/square displacement, purple
  dash-dot/triangle SVF and larger black diamond bounded-two atlambda1 only.
  Four-method single-row legend and lighter0.12-alpha SD ellipses; no curve
  smoothing, lambda filtering, or metric changes.
- Four new selection/validation/style/layout tests plus prior17 tests passed.
  Visually inspected F/G4000×2200 PNGs; all labels/legend/points are visible.
  Independent saved-file verification reproduced400 mean/SD statistics from
  400 displayed records/350 unique source records, verified40 settings,12 axes,
  original metric equality, source hashes, PNG/SVG/PDF structure and fonts.
- Host pdfunite appended new F/G to original A–E without rerendering A–E.
  Extracted text for the first five pages is identical to the original PDF.
  Both original F/G and the lambda1 table are hash-verified unchanged.
- Accepted2026-10-15 21:28:53UTC under
  ARTIFACT_ROOT/repo_experiments/oasis_seed3_first10_ncc9_method_family_arc_panels_v1/panels_FG_sitreg_v2.
  Separate render/verification sources preserve historical source hashes.
  CPU-only container execution as1009:1020. No staging or commits.
