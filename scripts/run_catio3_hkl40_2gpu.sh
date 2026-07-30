#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/.venv/bin/activate"

SITE="$ROOT/.venv/lib/python3.11/site-packages"
CUDA_LIBS="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_nvrtc/lib"

export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# hkl40 work units require more than Dask's four-worker 15.4 GiB memory
# budget. Two workers safely receive 30.7 GiB each; GPUs 0 and 2 also have
# this machine's full-width PCIe links.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export DASK_LOCAL_DIR="${DASK_LOCAL_DIR:-$ROOT/.dask-local}"
export MOSAIC_RESIDUAL_LATTICE_SCRATCH="${MOSAIC_RESIDUAL_LATTICE_SCRATCH:-$DASK_LOCAL_DIR/hkl40-lattice}"
export MOSAIC_GPU_ALLOW_MULTI_THREAD_WORKER=1
export MOSAIC_NUFFT_EXECUTION_POLICY=gpu-required
export MOSAIC_NUFFT_SLOTS_PER_WORKER="${MOSAIC_NUFFT_SLOTS_PER_WORKER:-1}"
export MOSAIC_DASK_GPU_RESOURCE="${MOSAIC_DASK_GPU_RESOURCE:-$MOSAIC_NUFFT_SLOTS_PER_WORKER}"
export MOSAIC_GPU_HEADROOM_GIB="${MOSAIC_GPU_HEADROOM_GIB:-4}"
export MOSAIC_SCATTERING_STAGE2_STREAMING=1
export MOSAIC_STREAMING_STAGE1_PARALLEL="${MOSAIC_STREAMING_STAGE1_PARALLEL:-2}"
export MOSAIC_RESIDUAL_PREFETCH_FACTOR=1
export MOSAIC_RESIDUAL_SHARD_GRID_BUDGET_BYTES="${MOSAIC_RESIDUAL_SHARD_GRID_BUDGET_BYTES:-4294967296}"
export MOSAIC_RESIDUAL_CHECKPOINT_CADENCE_BATCHES=8
export MOSAIC_SCATTERING_INTERVAL_PAYLOAD_CACHE_MAX_BYTES=268435456
export MOSAIC_STREAMING_PAYLOAD_MEMO_MAX_BYTES=268435456
export MOSAIC_RESIDUAL_RIFFT_PAYLOAD_CACHE_MAX_BYTES=2147483648
export MALLOC_TRIM_THRESHOLD_=67108864
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# A stale value of 40 turns this problem into hundreds of redundant shards.
# Let MOSAIC choose the adaptive shard size from the live memory budget.
unset MOSAIC_RESIDUAL_INTERVALS_PER_SHARD

mkdir -p "$DASK_LOCAL_DIR" "$MOSAIC_RESIDUAL_LATTICE_SCRATCH"
cd "$ROOT"

if (( $# > 0 )); then
    CASES=("$@")
else
    CASES=(all sphere rod rest)
fi

for CASE in "${CASES[@]}"; do
    echo ">>> CaTiO3 hkl40 2-GPU case: $CASE"
    python -m core.main \
        "examples/config_3D/displacement/run_parameters_hkl40_${CASE}.json"
done
