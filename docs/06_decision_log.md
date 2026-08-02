---
doc_type: decision_log
status: active
---

# Decision Log

## 2026-06-25: Promote `inrorspline` To Real Repo Root

Decision:

- `/home/hengjie/code_sync/inrorspline` is the real Git repo root.
- `starting_kit/` remains as source material for now.
- Root scaffold is created without `pyproject.toml` or a Python package/test scaffold.

Rationale:

- The immediate need is governance, documentation, protocol safety, and reproducible workflow structure before implementation or GPU runs.

## 2026-06-25: Preserve Original IDIR Bending-Energy Scale

Decision:

- Keep `idir_original` training BE on the public original IDIR scale.
- Do not reinterpret raw IDIR alpha values as directly comparable to SINR or
  DirectCP alpha values.
- Treat any normalized IDIR BE implementation as a clearly tagged future method
  variant, not as a replacement for `idir_original`.

Rationale:

- The original IDIR BE implementation averages second-derivative terms and then
  divides the result again by the sampled coordinate batch size.
- With the default IDIR coordinate batch size of 10,000, this downscales the
  reported training BE by 10,000 relative to the pre-divide mean BE.
- Preserving this behavior maintains comparability with original IDIR runs and
  explains why IDIR alpha scales can differ sharply from SINR/DirectCP alpha
  scales.
- Cross-method comparisons should rely on `alpha * training_reg_loss` and common
  posthoc regularity metrics rather than raw alpha labels alone.

## 2026-07-11: Consolidate LungCT Under DIR-Lab

Decision:

- Use `/data/$USER/datasets/lungCT/DIR-Lab` as the sole LungCT dataset root.
- Keep raw `4DCT` and `COPD` data under `raw/`.
- Keep native R231 masks and pTVReg-style 1 mm data as separately named
  preprocessing protocols: `lungmask_R231_native` and `pTVReg_iso1mm`.
- Store migration/checksum records outside the dataset tree under
  `/data/$USER/datasets/manifests/` and do not maintain compatibility symlinks.

Rationale:

- The former top-level raw trees were byte-identical duplicates.
- Named protocol directories make preprocessing assumptions explicit and keep
  raw data separate from derived artifacts.
- External integrity records allow the dataset root to remain clean without
  discarding provenance.

## Pending Decisions

- Final public dataset handles that avoid private host paths.
- Whether to add a Python package scaffold, and which tooling to use.
- Which OASIS alpha/LR grid to lock for the first seed-3 run.
