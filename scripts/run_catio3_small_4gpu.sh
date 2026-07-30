#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/.venv/bin/activate"

SITE="$ROOT/.venv/lib/python3.11/site-packages"
CUDA_LIBS="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_nvrtc/lib"

export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DASK_LOCAL_DIR="${DASK_LOCAL_DIR:-$ROOT/.dask-local}"
export MOSAIC_GPU_ALLOW_MULTI_THREAD_WORKER=1
export MOSAIC_NUFFT_EXECUTION_POLICY=gpu-required
# Two small NUFFT tasks may share each 24 GiB RTX 4090.  Both scheduler
# resources must match because GPU tasks request one slot from each pool.
export MOSAIC_NUFFT_SLOTS_PER_WORKER="${MOSAIC_NUFFT_SLOTS_PER_WORKER:-2}"
export MOSAIC_DASK_GPU_RESOURCE="${MOSAIC_DASK_GPU_RESOURCE:-$MOSAIC_NUFFT_SLOTS_PER_WORKER}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "$DASK_LOCAL_DIR"
cd "$ROOT"

if (( $# > 0 )); then
    CASES=("$@")
else
    CASES=(all sphere rod rest)
fi

for CASE in "${CASES[@]}"; do
    echo ">>> CaTiO3 small 4-GPU case: $CASE"
    python -m core.main \
        "examples/config_3D/displacement/run_parameters_small_${CASE}.json"
done
