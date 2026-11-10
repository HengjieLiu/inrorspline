# GaussianDIR runtime and commands

Use the configured Docker container and its artifact-writing user. The tested
image digest is `sha256:8a159ba5e7f438cf36ea10d74a4a45fac1b0250d7d70f0dba114eefcbe3416da`;
the registry reference is captured in the external runtime receipt.
All commands below run inside that container with working directory
`/workspace/repo` and the environment variable `${ARTIFACT_ROOT}` configured.

The isolated system-site-packages venv is
`${ARTIFACT_ROOT}/envs/gaussiandir_pytorch241_cu118_py310_v1`. It uses Python3.10,
torch2.4.1+cu118, torchvision0.19.1+cu118, torchaudio2.4.1+cu118,
PyTorch3D0.7.8, hydra-core1.3.2, omegaconf2.3.0, iopath0.1.10,
SimpleITK2.4.1 and pytest8.3.5. It inherits numpy/scipy/nibabel/surface-distance
from the image. Base torch2.6.0+cu118 is unchanged. W&B is unused.

The container lacks nvcc. PyTorch3D therefore uses the matching official binary
[py310/cu118/pyt241 archive](https://conda.anaconda.org/pytorch3d/linux-64/pytorch3d-0.7.8-py310_cu118_pyt241.tar.bz2),
SHA-256 `123353f1b9e3bbeab80670a2a9f9968a5156cacd20088601c86d332fd99a3e76`.
Only its validated `lib/python3.10/site-packages/` members were extracted into
the isolated venv; the base Python installation was not changed. This matches
the supported torch range in the upstream [installation instructions](https://github.com/facebookresearch/pytorch3d/blob/v0.7.8/INSTALL.md).
The external venv holds the original archive, `pytorch3d_receipt.json`, install
logs, `pip_freeze.txt`, and a `docker_receipt.json` from an inspected container.
Rebuilding elsewhere requires capturing that container's actual receipt.

Invoke [python_env.sh](../../scripts/experiments/gaussiandir/python_env.sh) with
`bash`. It puts the isolated torch/NCCL libraries before inherited NVIDIA
libraries in `LD_LIBRARY_PATH` and disables bytecode writes. It performs no
installation. `GAUSSIANDIR_ENV` can identify another validated isolated venv.

```bash
bash scripts/experiments/gaussiandir/python_env.sh -m pytest -q \
  -p no:cacheprovider tests/test_gaussiandir.py

# Native resource smoke; use a fresh --out for every attempt.
bash scripts/experiments/gaussiandir/python_env.sh \
  -m scripts.experiments.gaussiandir.run \
  --config experiments/2026-11-06_oasis_rank000_gaussiandir/protocol_v1.json \
  --variant paper_reconstruction --mode unsupervised --smoke \
  --out "$ARTIFACT_ROOT/runs/oasis_rank000_gaussiandir_v1/NEW_SMOKE/paper_reconstruction_unsupervised"

# Real endpoint: repeat independently for both variants and both objectives.
bash scripts/experiments/gaussiandir/python_env.sh \
  -m scripts.experiments.gaussiandir.run \
  --config experiments/2026-11-06_oasis_rank000_gaussiandir/protocol_v1.json \
  --variant paper_reconstruction --mode supervised \
  --out "$ARTIFACT_ROOT/runs/oasis_rank000_gaussiandir_v1/NEW_EXECUTION/paper_reconstruction_supervised"
```

Variants are `released_control` and `paper_reconstruction`; modes are
`supervised` and `unsupervised`. `--resume` requires an incomplete output with
identical source/config hashes. Completed outputs are never reused. A checkpoint
stores model/population, active optimizer, CPU/CUDA RNG, shuffled sampler state,
stage/iteration and charged time. Every run snapshots source bytes externally.

`evaluate` replays `--field identity`, `--field sitreg`, or a saved candidate
field with `--config` and a fresh `--out`. After both references are evaluated,
`preflight --config ... --references <reference-root>` validates historical
metrics and independent native/canonical warps. Native/canonical interpolation
roundoff is recorded, including nearest-neighbor ties; exported SITReg labels
must agree with direct source labels to the declared tolerance.

`report --config ... --run-root ... --reference-root ... --out ...` requires all
four non-smoke endpoints and writes the full metric tables, figures and generated
Markdown report externally. The source [review notebook](review.ipynb) reads
`GAUSSIANDIR_REVIEW_ROOT`, displays these outputs inline, and is executed only
as an external copy. Runtime artifacts and private paths must never enter Git.
# Convergence/resource follow-up

The approved settings are `protocol_convergence_v1.json`; the scientific
definition is [convergence_spec.md](convergence_spec.md). `study_queue` launches
three repeats of each short/anchor schedule, one startup measurement and one
convergence trajectory per variant, with empty-GPU and frozen-source checks.
It requires completed native smoke summaries and a numerical validation receipt.

```bash
bash scripts/experiments/gaussiandir/python_env.sh \
  -m scripts.experiments.gaussiandir.study_queue \
  --config experiments/2026-11-06_oasis_rank000_gaussiandir/protocol_convergence_v1.json \
  --smokes "$ARTIFACT_ROOT/runs/oasis_rank000_gaussiandir_convergence_v1/SMOKE_ID" \
  --validation "$ARTIFACT_ROOT/runs/oasis_rank000_gaussiandir_convergence_v1/SMOKE_ID/validation.json" \
  --gpu-uuid "$GPU_UUID" \
  --out "$ARTIFACT_ROOT/runs/oasis_rank000_gaussiandir_convergence_v1/EXECUTION_ID"
```

After all timed workers finish, `study_evaluate --config ... --run-root ...
--out ... --workers 4` scores every archived checkpoint and endpoint with the
canonical evaluator. Posthoc workers may use separate idle GPUs; they never
overlap timed registration. Byte-identical repeated fields may share metrics
only after SHA-256 verification. `study_report --review ... --references ...
--out ...` writes complete external tables, figures, Markdown and provenance.

Execute an external copy of [convergence_review.ipynb](convergence_review.ipynb)
with `GAUSSIANDIR_CONVERGENCE_REVIEW` pointing at that report directory. Keep the
source notebook output-free. Never report saved SITReg field loading as inference.
