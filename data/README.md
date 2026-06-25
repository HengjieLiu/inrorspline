# Data

Raw and derived datasets do not live in Git.

Expected external roots:

```bash
export DATA_ROOT="/data/$USER/inrorspline"
export BRAIN_DATA="$DATA_ROOT/brain"
export LUNGCT_DATA="$DATA_ROOT/lungct"
```

Allowed here:

- tiny synthetic/demo samples, if needed later
- review-safe manifests
- documentation of expected dataset layout

Not allowed here:

- raw OASIS, LungCT, 4DCT, COPD, or other medical images
- checkpoints
- generated deformation fields
- private server paths
