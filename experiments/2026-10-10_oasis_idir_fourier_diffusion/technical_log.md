# Fourier-input sampled-IDIR technical log

## Known limitations

Exploratory first10 tuning; signed sampled global NCC and analytic diffusion,
not local NCC9/full-volume FD. Fourier inputs alter fan-in/initialization;
native rules are preserved and audited, not silently variance-matched.
Fixed2500 updates and shared workloads limit convergence/runtime claims.

## Approval and audit

User approved the proposed bandwidth/diffusion/capacity-control experiment and
requested all four GPUs. One new worker per GPU, UID:GID1009:1020.
Initial observed free memory on GPUs0–3 was approximately38.6/47.4/43.3/47.4GiB;
existing GPU0/2 workloads remain untouched. The running container image digest
matches P53. The dataset and P53 results will be checked before smoke execution.

Implementation is isolated in new core/worker/controller files; P53 numerical
sources and accepted runs remain unchanged. The plain lambda1 cohort is reused
only after full validation. Dedicated Fourier/model/sampling RNG handling
prevents changing sampled locations as a side effect of encoding dimensions.

Optional followup semantics are fixed before results: require higher Dice and
no worse other four mean metrics versus both selected controls. Otherwise stop
after primary bracketing and report the trade-offs rather than expanding search.

Qualification, launch and measured timing will be recorded below.

## Qualification correction: diagnostic autograd ordering

All 15 new unit tests passed, and all 38 combined old/new IDIR tests passed.
The first four-GPU smoke completed but correctly failed the strict plain-control
reproduction gate: initial-state and all sampled-index hashes matched, but the
final normalized field differed by max0.144796. First four loss entries were
identical; update5 first differed by1.788e-7 and the trajectories then separated.

An unchanged legacy100 run on GPU3 exactly reproduced its historical result,
ruling out a general cross-GPU reproduction failure in that check. Moving the
new diagnostics from GPU to CPU BEFORE training did not fix the discrepancy.
Moving diagnostic autograd AFTER training, using an immutable copy of the
initial state, restored EXACT agreement in all100 data/regularization/total-loss
entries and the dense field (max difference0). The specific low-level reduction
ordering mechanism is not claimed; the intervention isolates pre-training
diagnostic autograd as the trigger. No acceptance tolerance was relaxed.

Retain v1 smokes, all diagnostic reruns and its initial source snapshot under
the original v1 artifact root. Production uses a fresh v2 root/source signature,
and reruns all four smokes before launch. No rejected smoke is reused as a
scientific result. No model, loss, lambda, LR, update budget or native SIREN
initialization rule was changed to resolve the audit failure.

## Fresh v2 qualification passed

- All38 old/new IDIR tests passed again after the correction.
- Four simultaneous v2 smokes passed on GPUs0/1/2/3. Native-control normalized
  DVF max difference versus P53 was exactly0. All35label rows per smoke were
  validated and all four models passed terminal-checkpoint reconstruction.
- Numerical original-XYZ derivative checks had max absolute errors from
  3.0e-11 to3.73e-8, below the unchanged tolerances.
- Reserved training memory was0.668/0.668/0.725/0.629GiB. Warm median update
  times were0.00947/0.01397/0.01313/0.01277seconds across the four slots.
- CPU initial-state diagnostics confirm the expected native fan-in effect:
  first pre-sine SD about1.94–1.95 for Fourier inputs versus11.63–11.86 for
  plain inputs. Initial data-gradient norms remained finite/nonzero. At high
  bandwidth16/lambda3, regularization-gradient norm was about20.4times the data
  norm; at bandwidth0.5/lambda0.3 it was0.0047times. This motivates the joint
  bandwidth/lambda test and boundary checks, not unrecorded loss normalization.
- Production was released after inspecting these diagnostics. P53 code,
  checkpoints and report files remain unmodified; no failed v1 result is mixed
  into v2. Measured production timing will be used for the final launch ETA.

## Production launch

Detached controller PID79480 launched the first four production workers at
2026-10-11 05:54:37 UTC (2026-10-10 22:54:37 PDT), physical GPUs0/1/2/3,
worker PIDs79620/79621/79622/79626. All four were observed progressing with
10000 samples/update and unchanged LR1e-4. Initial queue:230 new fits, with
automatic complete-cohort boundary checks and the documented followup gate.
Live state/logs remain in the v2 run root; tables/plots in its review root.

The first three validated full-length fits completed in43.85–47.59seconds
including per-label/regularity evaluation, with30.76–33.29seconds in training.
Each had exactly2500updates/25million training coordinates and0.666–0.668GiB
reserved training memory; full coordinate-sequence hashes matched P53.
Initial planning ETA is50–75minutes for230new fits under current load,
excluding any boundary extensions and the conditional70fit followup phase.
The artifact store had about12TiB free, sufficient for the initial field budget.
