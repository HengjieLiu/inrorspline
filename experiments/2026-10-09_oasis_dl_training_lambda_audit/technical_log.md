# OASIS DL training audit technical log

## Known limitations

Historical configs/checkpoints survive without an attached April2025 source
commit. Strict loading alone does not prove inference parity. Six-pair smokes
measure memory, not convergence or whole-run memory/runtime bounds. Historical
cuDNN benchmark=True/deterministic=False is an explicit protocol override.
Production lambda training is not authorized.

## Audit

Recovered notebook commands, local W&B configs, full200-epoch logs, checkpoint
optimizer states and native model/loss/data implementations. Confirmed LR1e-4,
Adam AMSGrad, batch1, seed42, native bidirectional update counts, NCC9 and
voxel diffusion. Dataset manifest verifies300/30/200 and disjoint subjects.
Reference epochs171/191 for VFA/SITReg.

Historical train+validation sums:VFA7.4702h and SITReg8.2932h/run; median
train+validation98.50+35.19s and100.97+47.36s/epoch respectively. These are
historical observations, not current guarantees.

## Smoke results

Both full-volume FP32 native training smokes completed sequentially on GPU3.
VFA performed12 Adam updates across six pairs; SITReg performed6 joint
bidirectional updates across six pairs. Both passed strict checkpoint loading,
expected parameter counts, finite gradients, actual AMSGrad-state allocation
and source-checkpoint SHA256 immutability. No trained checkpoints were saved.

| Network | Phase | Allocated peak GiB | Reserved peak GiB |
| --- | --- | ---: | ---: |
| VFA | Fresh weights, including startup | 36.42272 | 39.54492 |
| VFA | Trained lambda1 weights | 36.42273 | 39.54688 |
| SITReg | Fresh weights, including startup | 33.33553 | 35.94141 |
| SITReg | Trained lambda1 weights | 25.78954 | 28.70703 |

Polled GPU-wide use peaked at40.06055GiB for VFA and36.48047GiB for SITReg;
GPU3 was otherwise idle. Polling every0.2s can miss short hardware peaks,
whereas PyTorch peaks are allocator counters. No GPU OOM occurred. Current
PyTorch2.6.0+cu118, CUDA11.8, cuDNN90100; historical benchmark mode retained.
Effective TF32:cuDNN enabled, matrix multiplication disabled. CPU checks verify
current wavereg squared-NCC9 values/gradients exactly match the public legacy
implementation. Native Grad3d matches the explicit forward-valid component/
axis-mean squared-difference definition (value error0, gradient error2.33e-10).

VFA first-pair startup36.89s; subsequent pair timings about2.2..2.8s including
two optimizer updates and gradient checks. SITReg startup50.70s; trained
phase about1.78..1.81s/pair. These short diagnostics omit validation and file
loading and are not whole-run ETA measurements.

Outputs under the artifact key's`vfa_gpu3_attempt2` and
`sitreg_gpu3_attempt1`:manifest/source hashes, history, sampled GPU use,
stdout, result JSON and COMPLETED. VFA attempt1 failed before model execution
because TransMorph utilities required missing`pystrum`; it was not a GPU
failure. Installed historical`pystrum==0.4` into the new experiment's isolated
runtime_pydeps, without changing shared Python; preserved attempt1 directory.

## Proposed scheduling and remaining gates

Budget at least42GiB free for VFA and39GiB for SITReg, one worker/GPU, checking
live free memory before launch. These include headroom but are not guarantees
for all learned fields/lambdas. GPU3 was released after the smokes. Other
workloads remain untouched. At the audit snapshot only GPUs2/3 satisfy these
conservative launch budgets; GPUs0/1 must wait for sufficient free memory.

Eight fresh runs at historical7.47/8.29h correspond to approximately63GPU-hours
total, before evaluation and shared-resource overhead. Two eligible GPUs imply
roughly32+hours; four eligible GPUs roughly16+hours. No production ETA is a
promise. All full training remains approval-gated; saved-field inference parity
and a scoped adapter/update-contract test remain preproduction gates.
