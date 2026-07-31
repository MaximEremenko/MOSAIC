#!/bin/bash
# Per-rank wrapper for MULTI-NODE-PER-CASE runs (backend "mpi").
#
#   mpirun -np 6 --host node3:3,node1:1,node2:2 \
#       bash .../mpi-rank.sh /mnt/shared/configs/.../run_parameters.json
#
# dask-mpi SPMD shape: rank 0 becomes the scheduler, rank 1 continues as
# the MOSAIC driver, ranks >= 2 become workers. Every rank runs this same
# script; each node contributes one worker rank per GPU, and the rank's
# GPU is pinned here via CUDA_VISIBLE_DEVICES from the MPI local rank —
# exactly how a SLURM+MPI deployment maps ranks to cards.
set -euo pipefail

# ssh-launched remote ranks get a fresh login env (no compose vars):
# derive the software root from this script's own location on the shared
# stack, which is identical on every node.
ROOT="${MOSAIC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG="${1:?usage: mpi-rank.sh <run_parameters.json>}"

source "$ROOT/.venv/bin/activate"
SITE="$ROOT/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/mnt/shared/pylibs${PYTHONPATH:+:$PYTHONPATH}"

# One worker rank per GPU: pin this rank to one card by MPI local rank.
GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${GPUS:-0}" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(( ${OMPI_COMM_WORLD_LOCAL_RANK:-0} % GPUS ))
fi

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
