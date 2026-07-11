# Draft Paper Section: Optimization Path Versus Representation

These first-three-case diagnostics separate three mechanisms: whether updates move toward a strong reference deformation, whether SINR updates are a scalar/kernel variant of direct control-point updates, and whether hard LungCT cases require coarse-to-fine capture range.

The sampled SINR-vs-DirectCP cumulative early update similarity stayed below the exploratory gate in all three cases; the maximum observed similarity was `0.388`. This argues against the simple interpretation that SINR behaves like a rescaled single-stage DirectCP optimizer.

Case08 is the most diagnostic split: SINR cps4 can fit the pTVReg target field when supervised directly (best field-fit native snap TRE `1.043` mm), but its image-registration run remains poor (best native snap TRE `11.735` mm). In contrast, multistage DirectCP succeeds on the image-registration objective (best native snap TRE `1.002` mm).

The new schedule-ablation rows test whether the multistage rescue survives when the image pyramid is weakened or removed.
The best completed ablation so far is `ptv_nopyr_ds1-1-1-1_lr0p01` on case `1` with reg1mm snap TRE `1.076` mm.
On case01, no-pyramid control staging is already competitive in the exploratory schedule (best reg1mm snap TRE `1.076` mm).
On case08, the best completed no-pyramid row remains much worse (reg1mm snap TRE `6.478` mm), so control staging alone is not yet explaining the hard-case rescue.
The current best case08 partial-pyramid row has reg1mm snap TRE `11.937` mm.
Adding back the full standard `8,4,2,1` image pyramid gives a large recovery in the case08 LR probe (best reg1mm snap TRE `2.270` mm), but this short-budget result still leaves room for optimizer/runtime effects.
The LR/runtime refinement then closes the apparent pyramid gap: ptv_standard_lr0p01_e100 `1.213` mm, ptv_standard_lr0p01_e200 `1.217` mm, ptv_partial_lr0p01_e100 `1.219` mm, ptv_nopyr_lr0p01_e100 `1.218` mm. Under LR `0.01` with longer e100/e200 schedules, case08 converges near `1.21` mm even for partial and no-pyramid controls, so the mechanism is best described as optimizer scale and runtime interacting with capture range rather than a fixed architectural requirement for the image pyramid.

Interpretation: the current evidence favors an optimization-path and capture-range explanation over a representation-label explanation. This section should be framed as mechanism evidence on selected diagnostic cases, not as a population-final method table.
