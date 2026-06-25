# Data Inspection Notebooks

Run these notebooks inside the default Docker container:

```bash
docker exec -it py310sitreg_v1_test1 bash
cd /homebase/code_sync/inrorspline
```

The notebooks call the scripts under `scripts/data/inspection/` and
`scripts/data/preprocessing/`. Generated figures and reports belong under
ignored `outputs/` or external data roots, not Git.
