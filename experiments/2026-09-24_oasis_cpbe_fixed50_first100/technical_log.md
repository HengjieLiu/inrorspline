# Technical log

## Known limitations

- The first ten OASIS cases were used to select these candidates. The report
  therefore includes a ranks 10–99 sensitivity summary.
- CP-BE previously failed strict repeated-Adam numerical equivalence to the
  dense analytic BE implementation. Empirical registration quality does not
  remove that warning.
- Original-fast and 2500-step quality anchors are historical saved summaries.
  Their registration is not rerun in this experiment.
- Historical baseline timing is a controlled first10 GPU3 F/T reference, not a
  same-run OASIS-100 timing comparison.
- Parallel metric evaluation is offline and excluded from runtime.

## Execution record

- User selected new-candidates-only registration and automatic idle-GPU
  selection.
- User requested substantially parallel evaluation. The controller calibrates
  16 versus 32 persistent workers after all timed work completes.
- No paper, PDF, LaTeX, production default, or commit is part of this run.

## Completed run

- Run ID: `oasis_cpbe_fixed50_first100_20260925T052403Z_v1`.
- Registration used physical GPU 0. All 202 primary timing rows were accepted
  on their first attempt; no monitored timing interval was contaminated.
- Evaluation initially stopped before producing metrics because the ownership
  check matched both the Docker client and its Python child. The continuation
  restricted ownership matching to Python processes, recorded the amendment,
  and did not rerun or alter the frozen timing rows.
- The 16-worker pilot processed 32 cases at 0.278 cases/s. The 32-worker pilot
  processed 64 cases at 0.975 cases/s (3.506x the pilot throughput), with
  262.9 GiB host memory available and 15.78 GiB used on each GPU. The safety
  gates therefore retained 32 workers for the remaining 104 evaluations.
- Acceptance passed with 202 new timing rows, 200 new evaluation rows, and 100
  unique ranks for each of the two historical quality anchors.

## Primary results

| Method | Cold s | Warm s, mean +/- SD | Dice | HD95 mm | FG SDlogJ | FG folding % |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 50/50/50 + cache/batch | 14.725 | 1.918 +/- 0.127 | 0.80904 | 2.1705 | 0.8557 | 3.6279 |
| Fixed 50/50/50 + AMSGrad | 15.178 | 1.987 +/- 0.137 | 0.80902 | 2.1628 | 0.8558 | 3.6254 |
| Original fast analytic BE, historical | 21.617 | 9.154 +/- 0.338 | 0.80899 | 2.1815 | 0.8770 | 3.9035 |
| 2500-step analytic BE, historical | 74.322 | 57.725 +/- 0.142 | 0.80962 | 2.1677 | 0.8715 | 3.8623 |

The new timings contain 100 warm observations each on GPU 0. Historical timing
contains ten warm observations on GPU 3 and is contextual, while historical
quality covers the same OASIS ranks 0--99. Relative to original fast, the two
new methods differ in mean Dice by +0.000052 and +0.000034 and in mean HD95 by
-0.0110 mm and -0.0187 mm. Relative to the 2500-step anchor, their mean Dice
is lower by 0.000576 and 0.000594. Both CP-BE variants have lower foreground
SDlogJ and folding percentages than both analytic-BE anchors under the recorded
metric convention. These empirical results do not establish strict numerical
equivalence between CP-BE and dense analytic BE.

Generated reports and machine-readable tables remain outside Git under the run
and review roots specified in `codex_execution_spec.md`.
