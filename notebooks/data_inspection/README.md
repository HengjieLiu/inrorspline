# Data Inspection Notebooks

Run these notebooks inside the default Docker container:

```bash
docker exec -it py310sitreg_v1_test1 bash
cd /homebase/code_sync/inrorspline
```

The notebooks call the scripts under `scripts/data/inspection/` and
`scripts/data/preprocessing/`. Generated figures and reports belong under
ignored `outputs/` or external data roots, not Git.

- `oasis_brain_mask_inspection.ipynb`: OASIS foreground/mask visual audit.
- `lungct_4dct_mask_inspection.ipynb`: native 4DCT T00/T50 lung-mask audit.
- `lungct_4dct_ptvprep_visual_audit.ipynb`: pTVReg-index/trilinear 1 mm
  preprocessing audit with original/crop/resampled image-mask overlays and
  landmark verification.
