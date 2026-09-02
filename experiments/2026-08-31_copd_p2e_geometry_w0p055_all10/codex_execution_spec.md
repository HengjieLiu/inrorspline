---
doc_type: codex_execution_spec
status: planned
date_planned: 2026-08-31
---

# COPD P2e-geometry isoTV `0.055` all-ten-case candidate screen

## Purpose and boundary

Test whether the **4DCT-selected** pTV isoTV weight `0.055` transfers to the
completed COPD P2e geometry and schedule.  This is a separately named COPD
candidate screen.  It must not silently replace P2e (`w=0.11`) or update the
cross-dataset report before the result is reviewed.

## Frozen registration protocol

| Item | Value |
| --- | --- |
| Dataset/cases | canonical pTV-style 1-mm COPD, cases `01-10` |
| Direction | moving eBHCT to fixed iBHCT |
| Protocol tag | `ptv_landmark_crop_1mm_no_lungmask` |
| Method | direct cubic B-spline control points, no INR |
| Control stages | `64 -> 32 -> 16 -> 8 -> 4` |
| Image downsampling | `16 -> 8 -> 4 -> 2 -> 1` |
| Stage epochs | `400,400,400,400,900` (`2500` total) |
| Similarity | pTV local CC, `sigma=2.1 mm` |
| Regularizer | physical control-value isoTV, `csqrt=5e-3` |
| **Changed parameter** | **isoTV weight `0.055`** |
| Domain | five-voxel border mask; no lung-mask loss weighting |
| Optimizer | Adam, `lr=1e-3` |
| Seed | `1` |
| Stage transfer | exact crop-aligned dyadic cubic B-spline knot insertion |
| Saved output | final dense displacement field and standard case summary |

The only intended scientific difference from COPD P2e is `ptv_isotv_weight`:
`0.11 -> 0.055`.  The run name, JSON summary, and review artefacts must retain
the candidate label and literal weight.

## Source provenance

- Repository commit: `8302f7b9d244` on branch `gpu8`, with unrelated dirty
  worktree changes present.
- Launcher SHA-256:
  `cd63a82d7cba2edfcf9da3c7a71e3307b3888dea6baabef92984a675a2eeb403`.
- Per-case runner SHA-256:
  `d77809ff8366a13420c8cf29c6793fd610cd011d5e0356da3ac851bc453614c4`.
- Aggregator SHA-256:
  `146c1c8f4b8e4281a28ee86f146c746394f66a5e75e7dc38900a4a78e27c3970`.
- Container: `py310sitreg_v1_test1`
  (`hengjieliu96/dl_dir:py310sitreg_v1`).
- Inputs: `${LUNGCT_PTVREG_ISO1MM}`; no raw data are copied into this
  repository.

## Execution

Use only GPU `0`.  The run root is
`${SCRATCH_ROOT}/runs/copd_p2e_geometry_w0p055_all10_v1`; the review root is
this directory.  First execute a five-stage, one-epoch-per-stage case-01
smoke in a distinct `smoke/` run root.  Only after it validates the requested
weight and writes finite field/TRE outputs may the ten full cases be launched
sequentially on GPU `0`.

The full launcher command must explicitly set:

```text
--gpus 0
--stage-full-cps 64,32,16,8,4
--stage-downsample 16,8,4,2,1
--stage-epochs 400,400,400,400,900
--ptv-isotv-weight 0.055
--seed 1
--lr 1e-3
```

## Acceptance and reporting

- Confirm the smoke records all frozen settings, finite losses, finite native
  and registration-grid TREs, and a final dense field with the expected grid.
- Require `10/10` completed full-case summaries before cohort aggregation.
- Verify every case summary has `ptv_isotv_weight=0.055`, the frozen stage
  lists, and finite headline TRE.
- Generate a self-contained candidate `review_report.md`, case rows, mean row,
  JSON summaries, and a command/provenance manifest in this folder.
- The existing aggregator uses population SD for its aggregate CSV; the review
  must label that fact.  If a cross-report comparison is requested later,
  recompute sample SD (`ddof=1`) from the ten case means, as in the current
  comparison report.
