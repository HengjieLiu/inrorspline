# Execution specification: FireANTs CT-only two-window audit

## Entry points

- `scripts/experiments/learn2breath_fireants_greedy/run_pairlocal.py`
- `scripts/experiments/learn2breath_fireants_greedy/run_all_two_window_audit.py`
- `scripts/experiments/learn2breath_fireants_greedy/render_comparison_audit.py`

The runner imports the established P2e pair loader/evaluators but does not
invoke the P2e optimizer. It constructs only two scalar, in-memory normalized
CT `Image` objects for FireANTs. The output contract and cohort evaluations
are intentionally shared.

## External runtime contract

The FireANTs Docker container mounts the repositories and datasets under its
legacy paths. Provide all locations as runtime variables; generated fields,
per-case JSON/CSV artifacts, and logs remain outside Git.

```bash
RUN_ROOT=...                 # new external FireANTs output root
PAIRLOCAL_ROOT=...           # COPD pair-local root
VALIDATION_ROOT=...          # Learn2Breath train/validation root
R231_ROOT=...                # prior R231 mask-directory child
L2B_REPO=...                 # Learn2Breath repository for challenge evaluator
P2E_RESULTS_ROOT=...         # completed P2e all20 audit root, read-only
REPO_ROOT=/homebase/code_sync/inrorspline
SCRIPT=$REPO_ROOT/scripts/experiments/learn2breath_fireants_greedy/run_all_two_window_audit.py
```

Create the immutable manifest, preflight all 40 rows, then launch the four
persistent workers and aggregate:

```bash
python "$SCRIPT" make-manifest --out-root "$RUN_ROOT" \
  --copd-data-root "$PAIRLOCAL_ROOT" --validation-data-root "$VALIDATION_ROOT" \
  --r231-root "$R231_ROOT" --learn2breath-repo "$L2B_REPO" \
  --p2e-results-root "$P2E_RESULTS_ROOT" --gpus 0,1,2,3
python "$SCRIPT" preflight-all --manifest "$RUN_ROOT/manifest/fireants_audit_manifest.json" --out-root "$RUN_ROOT"
python "$SCRIPT" launch --manifest "$RUN_ROOT/manifest/fireants_audit_manifest.json" --out-root "$RUN_ROOT"
python "$SCRIPT" aggregate-fireants --manifest "$RUN_ROOT/manifest/fireants_audit_manifest.json" --out-root "$RUN_ROOT"
python "$SCRIPT" aggregate-combined --manifest "$RUN_ROOT/manifest/fireants_audit_manifest.json" --out-root "$RUN_ROOT"
```

The combined report creates two detailed 40-row tables: COPD per-case TRE and
folding, and validation per-case Dice (including all five lobes) and folding.
It also records cold/warm, optimizer-only, full-case, worker-case, worker-total,
and makespan timing. P2e CUDA 11.8 and FireANTs CUDA 12.1 timings are labelled
by environment and must not be treated as a cross-container runtime ranking.

## Notebook execution contract

Run notebooks inside the FireANTs container after aggregation with the output
roots supplied only as environment variables. The notebooks render inline:

```bash
export L2B_FIREANTS_REPO_ROOT="$REPO_ROOT"
export L2B_FIREANTS_AUDIT_ROOT="$RUN_ROOT"
export L2B_FIREANTS_P2E_RESULTS_ROOT="$P2E_RESULTS_ROOT"
export L2B_FIREANTS_PAIRLOCAL_ROOT="$PAIRLOCAL_ROOT"
export L2B_FIREANTS_R231_ROOT="$R231_ROOT"
export L2B_FIREANTS_VALIDATION_ROOT="$VALIDATION_ROOT"
export L2B_FIREANTS_L2B_REPO="$L2B_REPO"
jupyter nbconvert --to notebook --execute --inplace \
  "$REPO_ROOT/experiments/2026-08-30_learn2breath_fireants_greedy_two_window_audit/copd_comparison_audit.ipynb"
jupyter nbconvert --to notebook --execute --inplace \
  "$REPO_ROOT/experiments/2026-08-30_learn2breath_fireants_greedy_two_window_audit/validation_comparison_audit.ipynb"
```
