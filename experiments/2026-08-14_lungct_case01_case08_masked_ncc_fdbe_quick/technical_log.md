# LungCT case01/case08 masked-NCC FD-BE quick A/B technical log

## Known Limitations

- This is a two-case quick check on DIR-Lab 4DCT case01 and case08, not a
  population result.
- The comparison isolates the NCC data-loss domain within the FD-BE family:
  prior rows use full-crop global NCC, and new rows use fixed T00 lung-mask
  global NCC. It is not a native-IDIR equivalence claim.
- The regularizer remains dense forward-difference bending energy and is not
  masked. This is intentional so the FD-BE regularizer is matched to the prior
  unmasked runs.
- The run uses the regenerated pTVReg isotropic-1mm landmark-crop data, not
  native full-volume LungCT.
- Moving/T50 masks are not used for optimization.
- The code is launched from the current working tree after local quick-A/B
  method additions; record the exact diff before committing or promoting
  results.

## Protocol

- Dataset: DIR-Lab 4DCT.
- Cases: case01 and case08.
- Direction: T50/end-exhale moving image to T00/end-inhale fixed image.
- Data root: `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.
- Output root:
  `/database/inrorspline/runs/lungct_case01_case08_masked_ncc_fdbe_quick_v1`.
- Comparison root:
  `/database/inrorspline/runs/dirlab_case01_case08_panelA_lungmask_diag_curves_v1`.
- Primary headline TRE: native/crop snap-to-voxel TRE.
- Regularity: foreground registration-crop stdLogJ and nonpositive Jacobian
  percentage from the shared posthoc dense-field pipeline.

## Planned Jobs

| Method | Implementation | GPU | Cases | Hyperparameters |
|---|---|---:|---|---|
| INR-BSCP | `sinr_cps4_masked_ncc_fd_be` | 1 | 01, 08 | cps4, lr `1e-4`, alpha `10000`, epochs `2500` |
| D-BSCP | `direct_cp_cps4_masked_ncc_fd_be` | 2 | 01, 08 | cps4, lr `1e-3`, alpha `10000`, epochs `2500` |
| MR-D-BSCP | `direct_cp_32_16_8_4_masked_ncc_fd_be` | 3 | 01, 08 | cps `32,16,8,4`, downsample `8,4,2,1`, epochs `400,400,800,900`, lr `1e-3`, alpha `10000` |

## Execution Notes

- `py_compile`, smoke tests, launcher dry-run, and GPU preflight must pass
  before full launch.
- If `nvidia-smi` hangs or any assigned GPU lacks the configured free-memory
  margin, do not launch the full run.

## Completion Notes

- Full launch completed all six planned jobs with no failed records.
- Aggregation completed with six masked rows and six matched prior unmasked
  rows.
- Generated review files:
  - `masked_vs_unmasked_summary.csv`
  - `masked_vs_unmasked_long.csv`
  - `masked_vs_unmasked_report.md`
  - `masked_vs_unmasked_manifest.json`
- Quick-check outcome: fixed-T00 lung-mask NCC improved the fixed-mask
  `1-NCC` diagnostic for every method/case pair, but usually worsened TRE and
  foreground regularity at the reused unmasked hyperparameters. The only TRE
  improvement was case08 MR-D-BSCP (`1.087` mm to `1.054` mm), with stdLogJ
  increasing from `0.1922` to `0.3518`.
