# Figure Message

This figure is intended to show that fitting a known registration field and
solving the registration optimization problem are different tasks.

The paper-facing message:

- Direct field fitting can go very well for local-control spline models;
  DirectCP is especially strong when the target is SITReg on OASIS or pTVReg on
  LungCT.
- Registration behavior is not determined by target-field fit capacity alone.
  The similarity objective, representation, regularization, and optimization
  path all matter.
- LungCT case01 is an easier respiratory-motion example where several methods
  can make progress.
- LungCT case08 is harder: single-stage SINR/DirectCP cps4 have poor
  registration outcomes even though field fitting works, while multistage
  DirectCP gives the needed optimization path.

Curve definitions:

- OASIS registration: full-image global NCC data loss.
- LungCT registration: pTV LCC data loss on the border-masked pTVReg 1mm crop.
- Field fitting: masked target-field MSE, using SITReg for OASIS and pTVReg for
  LungCT.

IDIR reruns in this folder add missing matched similarity-eval curves only.
They do not change the original IDIR training objectives.
