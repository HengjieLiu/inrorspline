#!/usr/bin/env bash
# Invoke inside the configured Docker container. Never installs into base Python.
set -euo pipefail
GD_ENV="${GAUSSIANDIR_ENV:-${ARTIFACT_ROOT:?}/envs/gaussiandir_pytorch241_cu118_py310_v1}"
GD_BASE_SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
GD_LIBS="$GD_ENV/lib/python3.10/site-packages/torch/lib:$GD_ENV/lib/python3.10/site-packages/nvidia/nccl/lib"
for GD_LIBDIR in "$GD_BASE_SITE"/nvidia/*/lib; do
  if test -d "$GD_LIBDIR"; then GD_LIBS="$GD_LIBS:$GD_LIBDIR"; fi
done
export LD_LIBRARY_PATH="$GD_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "$GD_ENV/bin/python" -B "$@"
