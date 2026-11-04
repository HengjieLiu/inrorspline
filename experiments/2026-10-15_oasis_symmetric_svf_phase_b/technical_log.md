# P78 technical log

## Limitations

One exploratory tuned test case. Amplitude penalizes useful opposing motion as
well as shared motion; balance equalizes global RMS, not pointwise directions.
Unmasked midpoint NCC retains its foreground-reweighting incentive. Equal-split
diffusion equivalence is not universal nonlinear equivalence. Swap discrepancies
and numerical inverse/folding errors remain outcomes, not assumed guarantees.
Shared GPU timings are diagnostic. No Phase-A numerical source may change.

## Preflight

- User approved 28 fits and four smokes, all four GPUs available at first check.
- Container image unchanged; native sandbox nvidia-smi cannot access driver,
  but container query confirms four healthy RTX6000 Ada devices, each 48518MiB
  free. Use container snapshots for launch admission, not historical values.
- Artifact store has ample space; native-mm energies reuse dense branch fields.
- New isolated numerical module and joint loss-component histories implemented.

## Verification and launch

- All40 numerical/regression tests passed (12 Phase-B, 28 inherited regressions).
  Nonzero-field full-objective value/both-gradient exchange and zero-weight
  Phase-A parity pass. FP64 penalty gradients and native-unit reductions pass.
- Four full-size one-update-per-stage smokes and independent replays passed
  in63.80s, with no failures. GPU0/1/2/3 used baseline/amplitude/balance/combined.
- Numerical sources and selection controller hashes frozen before smokes.
  Production launched with the approved15->9->4 dependency graph. Report code
  remains separate from numerical sources; no Phase-A output/source changes.

## Completion (2026-10-16)

- All28 production fits and independent replays passed, no failures or retries.
  Production queue2416.742s (40.28min). Four smokes/replays passed;40 numerical
  plus5 reporting tests passed. Accepted report at07:27:41UTC (00:27:41PDT).
- Deterministic selection chose amplitude.001 and balance1 as single-term
  centers; combined grid(.0003,.001,.003)x(.3,1,3). All24 primary settings
  completed before choosing three swapped checks and the overall winner repeat.
  Amplitude and combined winners both usedGPU3, so their swap checks and the
  amplitude winner repeat ran sequentially there; balance swap usedGPU0.
- Best amplitude-only alpha.001,beta0: Dice.796250,HD951.980043mm,ASD.430644mm,
  FGSDlogJ.555949,fold0%. Fresh-A1 Dice.786141 and shared RMS27.316mm become
  .796250 and2.915mm. Winner branch RMS3.007/3.472mm,relative gap14.36%,
  native diffusion.018658/.019510; midpoint positives38.859/39.562%.
- Best balance-only beta1: .795014/2.034708/.444027,shared RMS18.552mm,
  branch RMS18.731/18.733mm. Best combined(.001,1): .793762/1.989542/.434541,
  shared2.877mm,branch3.220/3.221mm. Both best balance/combined have one
  canonical foreground folded voxel (4.646296e-05%); amplitude winner has zero.
- Repeat/swap do not modify selection. Winner repeat Dice.796218,swap.796538,
  but foreground field differences remain substantial locally: repeat EPE
  mean/p95/max .078744/.287850/11.501729mm; swap .077392/.289065/4.883435mm.
  Similar aggregate metrics are not identical fields. Numerical source/forward
  replay tests pass; no claim that CUDA alone explains the training divergence.
- Winner actual/best updates125/150/250 versus124/150/250, optimizer61.444s,
  worker121.366s,replay61.061s,allocated/reserved4.568/4.859GiB. Full experiment
  peak4.918GiB; admission8GiB sufficient. Timings remain diagnostic.
- Winner cycle mean forward/reverse .051508/.050119mm; foreground in-bounds
  coverage100%. Input-exchange differences and inverse-cycle errors are separate.
- Accepted report contains28 candidates,five historical references,selected
  comparisons,per-label/stage data,all28 loss histories/checkpoints,168 directional
  integration rows,140 saved candidate-statistic checks and output hashes.
  No later-phase optimization or first-ten expansion has been launched.
- Final read-only integrity pass verified all10 Phase-A and11 Phase-B accepted
  report file hashes, plus1025 saved CSV metric values across candidate,
  selected-comparison and integration-sensitivity tables. Git diff check clean.
