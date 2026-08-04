# COPD DUAL-INR six-arm execution specification

Run all ten COPD cases in the eBH -> iBH direction with seed 42. The 60 new
records are A0/A1/N0/N1/P0/P1 crossed with cases 01-10.

- A0 uses the released author-format five-lobe crop, sampling, adaptive
  schedule, and semantic-mask weight 0.1. A1 changes only the semantic weight
  to zero and replays its paired A0 transition manifest.
- N uses native full images and fixed-iBH R231 sampling. P uses the existing
  validated legacy-Python-bounds pTV-style crop at 1 mm and the exact
  five-voxel border interior, with no lung mask in optimization.
- N0/P0 use four full-resolution stages. N1/P1 use independently generated,
  antialiased factors [8,4,2,1]. Their total stage lengths are
  [400,400,800,900]; model weights persist while optimizer/scheduler state
  resets at stage boundaries.
- A TotalSegmentator five-lobe failure is technical, never replaced with
  two-lung R231 semantics. A poor TRE or regularity result is scientific output
  and never stops a case or causes tuning.

Before production, pass the all-case source/data dry-run and one case-01
technical smoke for each arm. Then run all A0 records, freeze their transition
traces, run all A1 records, and run independent N/P records through the
four-GPU queue. A technical failure receives at most one unchanged retry while
the remaining independent records continue; no scientific metric triggers a
retry or halt. Accept only finite histories, expected schedules, complete
fields, valid landmark transforms, and complete metric artifacts.

The final report has ten casewise rows for the six new arms plus official
pTVReg and MR-D-BSCP P2e. Official pTVReg is TRE-only with regularity N/A;
P2e is a whole-cohort reference selected by lowest completed population TRE.
All cross-preprocessing comparisons retain their protocol tags.
