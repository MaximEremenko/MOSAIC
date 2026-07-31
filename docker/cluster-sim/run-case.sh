#!/bin/bash
# Run one MOSAIC case INSIDE a sim node:
#   docker compose exec node2 bash /path/to/repo/docker/cluster-sim/run-case.sh \
#       /mnt/shared/configs/small_all/run_parameters.json
#
# Mirrors scripts/slurm/submit_hkl40_cases.sbatch: software from the
# read-only shared stack, node-local scratch, durable state + stage-1
# payload store on the shared filesystem. No device or core counts appear
# anywhere here — the node's allocation decides.
set -euo pipefail

ROOT="${MOSAIC_ROOT:?MOSAIC_ROOT not set (compose provides it)}"
CONFIG="${1:?usage: run-case.sh <run_parameters.json>}"

source "$ROOT/.venv/bin/activate"
SITE="$ROOT/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export DASK_LOCAL_DIR=/scratch/dask
export MOSAIC_RESIDUAL_LATTICE_SCRATCH=/scratch/lattice
export MOSAIC_WORKER_SCRATCH_ROOT=/scratch/worker
export MOSAIC_RESIDUAL_SHARD_SCRATCH_ROOT=/scratch/shard
mkdir -p "$DASK_LOCAL_DIR" /scratch/lattice /scratch/worker /scratch/shard
export MOSAIC_STREAMING_PAYLOAD_STORE="${MOSAIC_STREAMING_PAYLOAD_STORE:-/mnt/shared/stage1_store}"

export MOSAIC_NUFFT_EXECUTION_POLICY="${MOSAIC_NUFFT_EXECUTION_POLICY:-gpu-required}"
export MOSAIC_SCATTERING_STAGE2_STREAMING=1
export MOSAIC_GPU_ALLOW_MULTI_THREAD_WORKER=1
export MOSAIC_NUFFT_SLOTS_PER_WORKER=2
export MOSAIC_DASK_GPU_RESOURCE=2
export MOSAIC_RESIDUAL_PREFETCH_FACTOR=8
export MOSAIC_RESIDUAL_LATTICE_RAM_FRACTION=0.2
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=$((64<<20))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$ROOT"
exec python -m core.main "$CONFIG"
