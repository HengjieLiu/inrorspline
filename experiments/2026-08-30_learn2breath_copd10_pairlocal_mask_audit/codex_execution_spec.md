# Execution specification

This specification covers an inline-only visual audit. It is not an experiment
launch specification for registration or segmentation inference.

Run the notebook in the default Docker container from the repository mount:

```bash
docker exec py310sitreg_v1_test1 bash -lc '
  cd /homebase/code_sync/inrorspline &&
  LEARN2BREATH_PAIRLOCAL_ROOT=/database/datasets/lungCT/DIR-Lab/processed/dirlab_copd_lung250m4b_v1/challenge_like_1p5mm_224x192x224_pairlocal \\
  LEARN2BREATH_MASK_AUDIT_ROOT=/database/workspace/Learn2Breath26/experiments/2026-08-30_copd10_pairlocal_mask_audit/full/masks \\
  jupyter nbconvert --to notebook --execute --inplace \\
    experiments/2026-08-30_learn2breath_copd10_pairlocal_mask_audit/mask_audit.ipynb
'
```

The invocation must validate all 20 image grids and all 60 displayed source
masks before rendering. It must fail for missing files, incompatible NIfTI
shapes/affines, empty masks, unexpected lobe values, or an empty R231
pair-union coronal extent.

No `savefig`, output directory, image export, or source-data mutation is part
of this notebook. The only intended resulting artifact is the executed notebook
with inline tables and figures.
