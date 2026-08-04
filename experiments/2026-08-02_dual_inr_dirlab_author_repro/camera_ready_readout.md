# Camera-ready readout: DUAL-INR-DIR A0/A1

## Outcome

The released DUAL-INR-DIR implementation was faithfully reproduced on all ten
DIR-LAB 4DCT cases under its own author-code protocol. The schedule-controlled
semantic-loss ablation changed landmark TRE and deformation regularity only
slightly, while the objective-sensitive five-lobe Dice score decreased in all
ten cases when the semantic term was removed.

| Metric | A0 released-code full loss | A1 no semantic loss | A1 minus A0 |
|---|---:|---:|---:|
| Dense snap TRE (mm) | 0.9789 +/- 0.1727 | 0.9834 +/- 0.1669 | +0.0045 +/- 0.0139 |
| Dense subvoxel TRE (mm) | 1.1084 +/- 0.1319 | 1.1078 +/- 0.1316 | -0.0006 +/- 0.0033 |
| Five-lobe mean Dice | 0.96788 | 0.96463 | -0.00325 |
| Foreground diffusion | 0.0130 +/- 0.0071 | 0.0128 +/- 0.0070 | -0.0002 +/- 0.0001 |
| Foreground bending energy | 0.0046 +/- 0.0020 | 0.0045 +/- 0.0020 | -0.0001 +/- 0.0000 |
| Foreground SDlogJ | 0.1401 +/- 0.0361 | 0.1389 +/- 0.0353 | -0.0012 +/- 0.0015 |
| Foreground J <= 0 (%) | 0 | 0 | 0 |

Values are across-case mean +/- sample SD where an SD is shown. Paired deltas
are A1 minus A0. These are descriptive N=10, one-seed results; they are not an
equivalence or superiority test.

## Per-case headline TRE and 1 mm references

The pTVReg and MR-D-BSCP columns are contextual references using the same
original/native-landmark snap-to-voxel TRE convention. MR-D-BSCP uses one
cohort-level setting, `w=0.055`, selected because it has the lowest mean TRE in
the completed refined sweep; it is not a per-case oracle selection.

| Case | A0 (mm) | A1 (mm) | pTVReg 1 mm (mm) | MR-D-BSCP 1 mm, `w=0.055` (mm) | A1 minus A0 (mm) | A0 Dice | A1 Dice |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.7604 | 0.7677 | 0.7671 | 0.7926 | +0.0073 | 0.9751 | 0.9737 |
| 02 | 0.7123 | 0.7319 | 0.7509 | 0.7765 | +0.0196 | 0.9770 | 0.9757 |
| 03 | 0.8598 | 0.8759 | 0.9274 | 0.9484 | +0.0161 | 0.9687 | 0.9656 |
| 04 | 1.2575 | 1.2679 | 1.2657 | 1.2501 | +0.0105 | 0.9689 | 0.9653 |
| 05 | 1.1882 | 1.1649 | 1.0777 | 1.0897 | -0.0234 | 0.9708 | 0.9668 |
| 06 | 0.9433 | 0.9559 | 0.8316 | 0.8303 | +0.0125 | 0.9645 | 0.9623 |
| 07 | 0.9896 | 0.9781 | 0.8061 | 0.7960 | -0.0115 | 0.9657 | 0.9622 |
| 08 | 1.0890 | 1.1033 | 1.0142 | 1.0008 | +0.0143 | 0.9603 | 0.9557 |
| 09 | 1.0018 | 0.9952 | 0.9095 | 0.8817 | -0.0066 | 0.9625 | 0.9575 |
| 10 | 0.9874 | 0.9935 | 0.8410 | 0.8085 | +0.0060 | 0.9653 | 0.9614 |
| **Mean** | **0.9789** | **0.9834** | **0.9191** | **0.9175** | **+0.0045** | **0.9679** | **0.9646** |

The existing pTV-style source tables use population SD across the ten case
means: pTVReg is `0.9191 +/- 0.1525 mm`, and the lowest-TRE MR-D-BSCP sweep
point is `0.9175 +/- 0.1481 mm`. For consistency with the sample-SD convention
used for A0/A1 above, the corresponding re-aggregated SDs would be `0.1607`
and `0.1561 mm`, respectively. The submitted manuscript currently quotes the
more strongly regularized MR-D-BSCP `w=0.11` row (`0.9218 +/- 0.1523 mm`,
population SD), so the `w=0.055` choice must be identified explicitly if used
in the camera-ready revision. Because it was selected posthoc using cohort TRE,
it should be described as the lowest-TRE sweep reference, not as an unbiased
preregistered test estimate.

### Context regularity

| Method | Regularity evaluation grid/domain | FG diffusion | FG bending | FG SDlogJ | FG J <= 0 (%) |
|---|---|---:|---:|---:|---:|
| A0 released-code full loss | Native author crop; fixed T00 five-lobe union | 0.013041 +/- 0.007104 | 0.004576 +/- 0.001989 | 0.140097 +/- 0.036128 | 0.000000 +/- 0.000000 |
| A1 no semantic loss | Native author crop; fixed T00 five-lobe union | 0.012845 +/- 0.007011 | 0.004502 +/- 0.001964 | 0.138883 +/- 0.035288 | 0.000000 +/- 0.000000 |
| pTVReg 1 mm | pTV 1 mm crop; fixed T00 lung foreground, evaluation only | 0.011606 +/- 0.005294 | 0.008956 +/- 0.004134 | 0.197737 +/- 0.060408 | 0.027764 +/- 0.033850 |
| MR-D-BSCP 1 mm, `w=0.055` | pTV 1 mm crop; fixed T00 lung foreground, evaluation only | 0.014009 +/- 0.005789 | 0.005517 +/- 0.002261 | 0.223834 +/- 0.060733 | 0.036686 +/- 0.043095 |

Regularity entries are across-case mean +/- sample SD from the common posthoc
definitions; the two reference rows were re-aggregated from their per-case
records to match the A0/A1 SD convention. The pTV-style fixed lung foreground
is metric-only and was not used for optimization. Diffusion and bending are
voxel-coordinate finite-difference metrics, while SDlogJ and the
nonpositive-Jacobian percentage are dimensionless. Because DUAL and pTV-style
rows use different grids and foreground domains, these values provide
deformation-quality context but are not clean method-only regularity
comparisons.

### Preprocessing and provenance differences

- **DUAL A0/A1:** native spacing, per-image min-max normalization, a
  two-voxel-padded TotalSegmentator fixed/moving five-lobe-union crop, and
  fixed-lobe coordinate sampling. A0 also uses the released semantic lobe
  loss; A1 sets only that loss weight to zero.
- **pTVReg 1 mm:** fixed/moving landmark-union bounding box with margins
  `[x,y,z]=[10,10,5]`, CT clipping/scaling `[80,900] -> [0,1]`, approximately
  1 mm isotropic resampling, no lung-mask optimization, and a five-voxel
  border exclusion. The table uses the validated uploaded fields that produce
  the manuscript's `0.919 +/- 0.152 mm` row, then evaluates TRE after mapping
  the field back to native landmarks. The official repository's separately
  saved headline artifact is slightly different (`0.9176 mm` mean).
- **MR-D-BSCP 1 mm:** the same pTV-style landmark crop, intensity scaling,
  isotropic registration grid, and no-lung-mask optimization convention, with
  the `32-16-8-4` control schedule, `8-4-2-1` image schedule, local NCC, and
  isoTV. This is our PyTorch preprocessing/optimizer path, not the exact
  official pTVReg MATLAB implementation. The reference uses the single
  lowest-cohort-TRE refined-sweep setting `w=0.055` for all ten cases.

These protocol tags are why the 1 mm rows should remain references rather than
being presented as isolated effects of model architecture.

## Acceptance and fidelity

- All `20/20` records passed source, data, config, output, finite-history, and
  provenance gates.
- A0 mean TRE `0.97894 mm` reproduces the reported `0.98 mm` author mean and
  passes the preregistered `0.82-1.12 mm` fidelity interval.
- Every A1 record exactly replays its case-specific A0 multiscale boundaries.
- Direct-INR and decoded-dense snap TRE agree to within the locked tolerance;
  the pilot differences were exactly zero.
- Visual review of small-motion case01 and large-motion case08 found no
  apparent sign, orientation, crop, lobe-label, or schedule defect.

## Camera-ready interpretation

The clean statement supported by this experiment is:

> Under the released DUAL-INR-DIR author-code protocol, the method reproduces
> its reported DIR-LAB accuracy. Removing its semantic lobe-loss term while
> holding the realized multiscale schedule fixed has little descriptive effect
> on landmark TRE or common deformation regularity, although lobe overlap is
> consistently lower without that objective term.

This is useful as a compact additional baseline/ablation for the camera-ready
manuscript because it addresses the omitted public Dual-INR comparison and
tests whether its semantic loss, rather than its dual/multiscale optimization,
explains landmark accuracy. It should not be presented as proof that the
semantic term has no effect, or as an equivalence result.

Keep this result in a separate author-protocol block or appendix table. It is
not a clean method-only comparison with the manuscript's pTVReg-style 1 mm
isotropic rows: A0/A1 use native spacing, a TotalSegmentator fixed/moving
five-lobe-union crop, fixed-lobe sampling, sampled global NCC, and the released
regularizers. A1 still uses masks for crop and sampling and must not be called
"no mask."

Also disclose that the released code implements the semantic objective as a
summed per-lobe MSE, whereas the DUAL paper describes BCE. Here, "A0 original"
means the pinned released-code behavior.

For a one-week revision, the defensible stopping point is this verified A0/A1
result plus precise protocol tagging. Porting DUAL-INR to native full-volume
IDIR, pTVReg no-lung-mask/1 mm isotropic preprocessing, or an MR-D-BSCP-style
residual schedule would answer different controlled questions and should use
new experiment IDs rather than being folded into this reproduction after the
fact.

## Review files

- Detailed aggregate: `results/dual_author_report.md`
- Per-case records: `results/dual_author_per_case.csv`
- Paired deltas: `results/dual_author_paired_deltas.csv`
- Boundary audit: `results/dual_author_boundary_audit.csv`
- Common regularity provenance: `regularity/posthoc_author_regularity_manifest.json`
- Pilot visual audit: `pilot_review/author_review_manifest.json`
- pTVReg reference cases and regularity:
  `../2026-06-26_lungct_directcp4stage_midratio_pareto_extension/pareto_visual_review_midratio_extension/combined_case_rows_with_ptvreg.csv`
- MR-D-BSCP refined-sweep cases:
  `../2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/directcp_refined_case_rows.csv`
- MR-D-BSCP refined-sweep aggregates:
  `../2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/directcp_refined_mean_rows.csv`
