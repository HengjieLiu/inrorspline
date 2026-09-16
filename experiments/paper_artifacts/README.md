# Final-Paper Artifact Standards

This folder is the source of truth for final-paper builders, display names,
and hyperparameter-variant selection. Generated figures, tables, manifests,
reports, and executed review notebooks are external-only.

Set `ARTIFACT_ROOT` before running a supported builder. Generated paper
artifacts belong under
`${ARTIFACT_ROOT}/repo_experiments/paper_artifacts/<artifact>/<version>/`.
The repository retains only source code, standards, nickname maps, and
output-free notebook sources. Historical versioned paths in registry and paper
documents are archive-relative references, not repo-local output destinations.

## Files

- `artifact_standard.json`: canonical LR policy, artifact versions, selected
  hypervariants, temporary exceptions, and TODOs.
- `method_nicknames.csv`: editable method display-name map. Edit the
  `nickname` column only when changing paper-facing method names.
- builder scripts: source-only generators that validate output roots through
  `ARTIFACT_ROOT`.

## Output Routing

The standard JSON and nickname CSV are Git-managed source configuration.
Generated tables, figures, manifests, reports, and executed notebooks belong
outside Git. Set `ARTIFACT_ROOT`; the current Pareto builder defaults to:

```text
${ARTIFACT_ROOT}/repo_experiments/paper_artifacts/pareto_figure/<current-version>/
```

An explicit `--review-root` is allowed only when it is an absolute path inside
`${ARTIFACT_ROOT}/repo_experiments/`. The `review_key` values in
`artifact_standard*.json` define those logical destinations; legacy archived
manifests may still contain historical repo-relative `output_dir` values.

The current v012/v013 Pareto builders and the motion-pattern builder require
explicit absolute paths for prior generated inputs outside the Git worktree.
Use their `--help` output with archived paths under the external store; they do
not revive deleted `experiments/` output trees as implicit inputs.

The standard-driven Pareto builder likewise requires `--oasis-source`,
`--lung-mean-source`, and `--lung-no-sweep-source`. These are archived
generated tables and must be explicit absolute external inputs; its output
still defaults to the configured external `review_key`.

## Current LR Standard

- IDIR: `1e-4`
- SINR: `1e-4`
- DirectCP: `1e-3`
- DirectCP multistage: `1e-3`

## Final-Control-Spacing Standard

For the merged Panel A/B fitting figure, compare single-stage spline methods at
the same final control spacing as the multiresolution DirectCP curve whenever
the data are available. For OASIS, `MR-D-BSCP` uses the `8-4-2` schedule and
therefore ends at `cps2`; the matched Panel A/B comparison uses OASIS
`INR-BSCP`/`D-BSCP` cps2 curves. For LungCT, `MR-D-BSCP` uses the
`32-16-8-4` schedule and ends at `cps4`; the existing LungCT
`INR-BSCP`/`D-BSCP` cps4 curves are already final-spacing matched.

For OASIS Pareto figures and best-result tables, cps2 is the matched main
comparison against `MR-D-BSCP (8-4-2)`. cps4 rows may remain as
spacing-ablation or supplemental context, but should be labeled with explicit
`(cps4)` suffixes.

The current OASIS DirectCP 8-4-2 row remains a documented v001 exception at
`lr=1e-2` because the canonical `lr=1e-3` rerun is not yet available. Replace
that row after the rerun is complete.

## Nickname Workflow

Generators should read `method_nicknames.csv` and use `nickname` for labels.
The default nickname is the current original name, so the first standardized
rerun is label-preserving. Future paper-name changes should happen in this CSV
instead of being hard-coded in plotting or table scripts.

For final paper Pareto figures and best-result tables, OASIS labels emphasize
the model family plus B-spline control spacing, for example
`INR-BSCP (cps2)`, `D-BSCP (cps4)`, and `MR-D-BSCP (8-4-2)`. LungCT labels
separate parameterization/model from optimization-loss family. In those labels,
`IDIR loss` is the paper-facing shorthand for the IDIR-style/global-NCC+BE
family used by those runs, and `pTVReg loss` denotes the pTVReg-style
LCC+isoTV objective family or pTVReg reference.

## Pareto Figure Metric Scope

- OASIS Pareto artifacts keep the full configured accuracy-regularity grid.
- OASIS Pareto `v001` writes two plot trees: an annotated review tree with
  alpha/reference-point labels, and a paper-facing no-annotation tree that
  removes alpha labels and reference text labels such as Greedy, VFA, and
  SITReg because those methods are already identified in the legend.
- LungCT Pareto artifacts use original snap TRE as the only accuracy metric,
  plotted against each configured regularity metric. Do not generate LungCT
  original-subvoxel, reg1mm-snap, or reg1mm-subvoxel Pareto panels for the
  paper-facing `v001` package.
- LungCT Pareto `v001` also writes annotated review plots and paper-facing
  no-annotation plots. The no-annotation plots remove ratio labels and pTVReg
  point text because the plotted methods are already identified in the legend.
