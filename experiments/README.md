# Experiments

`experiments/` is source-only. Create one directory per real experiment:

```text
experiments/YYYY-MM-DD_short-slug/
├── codex_execution_spec.md
├── technical_log.md
└── notebook_source.ipynb        # optional; no cell outputs
```

Keep hand-authored specifications, technical logs, configurations, and
unexecuted notebook sources here. All generated material belongs under
`${ARTIFACT_ROOT}`: run outputs, checkpoints, fields, logs, figures, PDFs,
CSV/JSON tables, generated reports, monitor files, and executed notebook
copies. Use `${ARTIFACT_ROOT}/repo_experiments/<experiment-key>/` for review
artifacts and `${ARTIFACT_ROOT}/runs/<run-id>/` for run artifacts.

Do not add a broad `.gitignore` rule for this directory. An accidental local
output should remain visible, then be moved to the artifact store and cleared
from the worktree before committing.
