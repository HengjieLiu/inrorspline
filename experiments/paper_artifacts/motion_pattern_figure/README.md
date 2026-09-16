# Motion Pattern Figure Artifacts

This artifact family builds reference-field coronal panels for paper figure
preparation. The v001 generator uses OASIS rank 9 SITReg and LungCT case01/case08
pTVReg fields, with per-case orientation notes in the manifest.

It writes to `${ARTIFACT_ROOT}/repo_experiments/paper_artifacts/motion_pattern_figure/v001/`
by default (or a compliant `--review-root`). Supply `--sitreg-root` and
`--ptvreg-root` explicitly as absolute external directories; the source-only
worktree no longer provides generated reference-field inputs.
