# Fit-Modes Figure Interpretation Review

This note translates the current working interpretation into paper-facing
language, then critiques it against the latest comparison figure and its
supporting tables:

- `fit_modes_method_comparison_curves.png`
- `fit_modes_method_comparison_curves.pdf`
- `fit_modes_method_comparison_selected_runs.csv`
- `fit_modes_method_comparison_curve_summary.csv`

The figure should be treated as a mechanistic illustration using one OASIS
case and two LungCT cases, not as the final population-level result.

## Cleaned User Interpretation

The experiment separates two related but different questions. In the
target-field fitting task, the deformation field is supplied by a strong
external registration method: SITReg for OASIS and pTVReg for LungCT. In that
setting, we ask whether each parameterization can reproduce the reference
field under a direct MSE objective. In the image-registration task, the target
deformation is latent: each method must discover a useful deformation by
optimizing image similarity and regularization.

The field-fitting results suggest that the local-control spline
parameterization has enough capacity for these reference fields. DirectCP cps4
reaches the lowest target-field MSE on OASIS rank-9 and on both LungCT cases.
On LungCT, the 4-stage DirectCP run reaches essentially the same final
finest-domain MSE as single-stage DirectCP cps4, which suggests that the final
cps4 representation is not the limiting factor for matching the pTVReg field.

The image-registration results tell a different story. On OASIS rank-9, SINR
cps4 and DirectCP cps4 both optimize the global NCC objective well and reach
better Dice than IDIR. On LungCT case01, several methods make reasonable
progress, but the relationship between image loss and TRE is already imperfect.
On LungCT case08, single-stage cps4 methods can reduce the pTV LCC loss but
still produce poor landmark TRE, while the 4-stage DirectCP schedule recovers
much stronger registration performance. This supports the view that LungCT
case08 needs a coarse-to-fine capture path, not merely a flexible final
parameterization.

IDIR complicates the story in a useful way. It is not the best field-fitting
model in these runs, and its matched LungCT pTV LCC diagnostic is not best
either. However, its LungCT TRE can be competitive. That means lower matched
image-similarity loss, lower target-field MSE, and better landmark TRE are not
the same endpoint. The method's sampling domain, regularization, coordinate
network prior, and the evaluation metric can rank methods differently.

## What Is Supported

- DirectCP cps4 is very strong for target-field fitting in this figure. Final
  masked normalized-grid MSE is `7.78e-06` on OASIS/SITReg, `2.85e-08` on
  LungCT case01/pTVReg, and `9.76e-08` on LungCT case08/pTVReg.
- LungCT 4-stage DirectCP has essentially the same field-fitting endpoint as
  single-stage DirectCP cps4: `2.85e-08` on case01 and `9.68e-08` on case08.
  This supports a capacity-vs-optimization distinction.
- OASIS rank-9 is comparatively forgiving in this selected-case figure:
  registration Dice is `0.8296` for SINR cps4, `0.8268` for DirectCP cps4, and
  `0.8110` for IDIR.
- LungCT case08 supports the capture-range interpretation. The selected
  registration TRE is poor for single-stage SINR cps4 (`11.73 mm`) and
  DirectCP cps4 (`13.81 mm`), but much better for 4-stage DirectCP
  (`1.00 mm`) and IDIR (`1.12 mm`).
- The figure supports the broader claim that registration behavior is not
  determined by representational capacity alone. The optimization path,
  similarity metric, regularization, and motion regime matter.

## What Needs Qualification

- Do not say that B-splines can fit "any" deformation. The supported statement
  is narrower: the cps4 local-control spline is expressive enough to fit these
  strong reference fields to very low MSE under direct optimization.
- Do not call SITReg or pTVReg fields ground truth. They are strong external
  reference fields, useful for probing capacity and optimization, but they are
  not the true physical deformation.
- Do not call IDIR simply bad. In this figure IDIR is worse for direct
  target-field MSE and worse on the matched LungCT pTV LCC diagnostic, but it
  can still achieve competitive LungCT TRE.
- Do not present this figure as a full statistical comparison. It uses one
  OASIS case and two LungCT cases with selected learning rates. It is best used
  to explain mechanisms that should then be supported by larger sweeps.
- Be careful with "case08 is harder." The figure strongly supports that case08
  is harder for these single-stage cps4 registration runs. If we describe it
  as a larger-motion case, the paper should cite the motion-regime statistics
  or a separate quantitative displacement/TRE characterization.

## Where The Interpretation Needs Correction

The main correction is that image loss is not the same thing as registration
success. LungCT case01 shows this already: single-stage DirectCP cps4 reaches a
low final pTV LCC loss (`2.63e6`, plotted as `2.63` after scaling), close to
SINR cps4 and 4-stage DirectCP, but its selected TRE is worse (`1.13 mm`
versus about `0.78 mm` for IDIR, SINR, and 4-stage DirectCP). So "DirectCP
fits the images" is not sufficient; the deformation can satisfy the image
metric while missing landmarks or producing a less useful geometric solution.

LungCT case08 makes the same point more dramatically. Single-stage DirectCP
cps4 ends with a slightly lower pTV LCC diagnostic than IDIR (`4.996e6` versus
`5.033e6`), but its selected TRE is much worse (`13.81 mm` versus `1.12 mm`).
The 4-stage DirectCP run improves the pTV LCC diagnostic further and also
recovers a much better TRE (`1.00 mm`). Therefore the paper should not argue
that the image-similarity curve alone explains all performance. It should say
that the image objective is necessary but not sufficient, and that the
optimization path changes which basin of deformation solutions is reached.

The other correction is about SINR in target-field fitting. It is not as strong
as DirectCP, but it is not uniformly poor. On OASIS/SITReg its final MSE is
close to DirectCP (`8.62e-06` versus `7.78e-06`). On LungCT it is clearly worse
than DirectCP but still better than IDIR in these field-fitting runs. The more
precise statement is that direct control-point optimization is the most
efficient and accurate target-field fit among the tested methods, while SINR
retains useful spline capacity but is not the best way to optimize a known
field under MSE.

## Paper Wording

Possible Results/Discussion paragraph:

> To separate representation capacity from registration optimization, we
> directly fit strong reference deformation fields using the same model
> families. In this target-field fitting setting, DirectCP cps4 reaches very
> low masked normalized-grid MSE for both OASIS/SITReg and LungCT/pTVReg
> targets, and the LungCT 4-stage DirectCP run reaches essentially the same
> final finest-domain MSE as single-stage DirectCP. Thus, for these examples,
> the final local-control spline representation is sufficiently expressive to
> reproduce the reference fields; failure in image registration cannot be
> explained by a simple lack of deformation-field capacity.

Possible follow-up paragraph:

> Image-based registration exposes a different limitation. On OASIS rank-9,
> SINR cps4 and DirectCP cps4 both optimize the global NCC objective well and
> outperform IDIR in Dice. On LungCT case08, however, single-stage cps4
> optimization reduces the pTV LCC loss but yields poor landmark TRE, whereas
> the coarse-to-fine DirectCP schedule substantially improves both the matched
> similarity diagnostic and TRE. This indicates that the effective deformation
> prior is induced jointly by the parameterization, regularization, similarity
> metric, sampling domain, and optimization path, not by the final
> representation alone.

Possible IDIR paragraph:

> IDIR provides an important caution for interpreting these curves. Its
> matched pTV LCC diagnostic is not best on LungCT, yet its TRE can be
> competitive. This discrepancy suggests that landmark accuracy, dense
> target-field MSE, and image-similarity loss emphasize different aspects of
> the solution. Coordinate-network priors, lung-mask sampling, and regularity
> may guide IDIR toward geometrically useful deformations even when the
> matched image-similarity diagnostic is not the lowest.

Concise figure-caption wording:

> Direct target-field fitting shows that cps4 local-control splines can
> reproduce strong SITReg/pTVReg reference fields to low MSE, especially under
> direct control-point optimization. Image registration behaves differently:
> on the harder LungCT case08, single-stage cps4 methods fit the image metric
> poorly in terms of TRE, while a coarse-to-fine DirectCP schedule recovers a
> useful solution. The figure therefore separates deformation-field capacity
> from the optimization and metric effects that determine registration
> performance.

## Recommended Paper Claim

The strongest safe claim is:

> The final spline parameterization has enough capacity for the tested
> reference fields, but registration performance depends on the effective
> deformation prior created by the image objective, regularization, sampling
> domain, and optimization path. This dependence is mild on the illustrative
> OASIS case and strong on the large-motion LungCT case08, where multistage
> optimization is needed.

Avoid stronger claims such as:

- "B-splines can fit any deformation."
- "SINR fails because its representation is weak."
- "IDIR is worse than spline methods."
- "Image loss directly predicts TRE."
- "These three cases prove the population-level conclusion."
