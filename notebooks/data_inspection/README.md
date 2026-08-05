# Data Inspection Notebooks

Run these notebooks inside the default Docker container:

```bash
docker exec -it py310sitreg_v1_test1 bash
cd /homebase/code_sync/inrorspline
```

The notebooks call the scripts under `scripts/data/inspection/` and
`scripts/data/preprocessing/`. Generated figures and reports belong under
ignored `outputs/` or external data roots, not Git.

Before execution, configure `INRORSPLINE_REPO_ROOT` and
`LUNGCT_DIRLAB_ROOT`. The COPD case02 audit additionally requires
`COPD02_PTVPREP_CASE_DIR` and `COPD02_PTVREG_MAT`; these variables keep host
and container paths out of the notebooks and their saved outputs.

- `oasis_brain_mask_inspection.ipynb`: OASIS foreground/mask visual audit.
- `lungct_4dct_mask_inspection.ipynb`: native 4DCT T00/T50 lung-mask audit.
- `lungct_4dct_ptvprep_visual_audit.ipynb`: pTVReg-index/trilinear 1 mm
  preprocessing audit with original/crop/resampled image-mask overlays and
  landmark verification.
- `lungct_copd_mask_inspection.ipynb`: native COPD iBH/eBH R231 lung-mask
  audit for cases 01–10 with coronal overlays and landmark-in-mask checks.
- `lungct_copd_ptvprep_visual_audit.ipynb`: all-case COPD pTV-style crop and
  1 mm preprocessing audit with native/crop/resampled image-mask overlays and
  landmark verification.
- `lungct_copd_case02_ptvprep_visual_audit.ipynb`: COPD case02 pTV-style crop
  and 1 mm preprocessing audit, using the same radiology orientation convention
  as the 4DCT notebook plus pTVReg outlier-landmark overlays. Re-execution
  requires `INRORSPLINE_REPO_ROOT`, `COPD02_PTVPREP_CASE_DIR`, and
  `COPD02_PTVREG_MAT` to point at the repo, COPD02 pTVPrep case folder, and
  selected pTVReg rerun deformation `.mat`.
