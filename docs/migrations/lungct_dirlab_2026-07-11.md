---
doc_type: data_migration
status: complete
---

# LungCT DIR-Lab Reorganization

On 2026-07-11, the local LungCT dataset was consolidated under the canonical
host root `/data/$USER/datasets/lungCT/DIR-Lab` and Docker root
`/database/datasets/lungCT/DIR-Lab`.

Current paths:

- Raw: `DIR-Lab/raw/{4DCT,COPD}`
- Native R231 masks: `DIR-Lab/processed/lungmask_R231_native/{4DCT,COPD}`
- pTVReg-style 1 mm data: `DIR-Lab/processed/pTVReg_iso1mm/{4DCT,COPD}`

Historical experiment specs and technical logs intentionally retain the paths
used at execution time. Their current pTVReg-equivalent root is
`/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`. The durable local
checksum and path-mapping record is outside the dataset tree at
`/data/$USER/datasets/manifests/lungCT_DIR-Lab_reorganization_2026-07-11`.
