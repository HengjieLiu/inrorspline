---
doc_type: literature_notes_index
status: active_draft
---

# Literature Notes

This folder contains review-safe, paraphrased reading notes derived from papers
stored privately under `private/literature/`. Do not commit downloaded PDFs or
raw extracted paper text unless their license explicitly permits redistribution.

## Notes

- `2022_wolterink_idir.md`: IDIR, continuous SIREN deformation fields.
- `2024_sideri_sinr.md`: SINR, INR-predicted FFD control-point displacements.
- `2026_gebauer_dual_inr_dir.md`: dual-branch multi-scale INR for thoracic CT.
- `2025_sideri_arc.md`: alignment-regularity characteristic evaluation.
- `2026_zeybekoglu_ptvreg_python.md`: accessible pTVReg Python implementation
  and COPD-relevant preprocessing notes.

## Code Links

- [external/IDIR](../../external/IDIR): official IDIR code.
- [external/SINR](../../external/SINR): official SINR code.
- [external/pTVreg](../../external/pTVreg): pTV/isoTV baseline code used in LungCT comparisons.
- [external/ptvreg-python](../../external/ptvreg-python): public Python
  pTVReg implementation used for the COPD fixed-parameter reproduction trial.
- [external/MIR](../../external/MIR): related registration benchmarking/utilities; not currently
  the official ARC anonymous-code release.

## Missing External Code

The dual-branch lung INR paper lists official code at
`https://github.com/IPMI-ICNS-UKE/DUAL-INR-DIR`. Add it with:

```text
[add git external] https://github.com/IPMI-ICNS-UKE/DUAL-INR-DIR
```

The ARC paper lists an anonymous code link. If that repository should be tracked,
add the de-anonymized or preferred canonical GitHub URL as an `external/`
submodule once selected.
