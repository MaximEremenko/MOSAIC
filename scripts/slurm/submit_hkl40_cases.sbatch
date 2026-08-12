#!/bin/bash
#SBATCH --job-name=mosaic-hkl40
#SBATCH --array=0-3
#SBATCH --gpus-per-node=4          # any count works: workers auto-size to it
#SBATCH --cpus-per-task=32
#SBATCH --mem=0                    # whole-node memory; MOSAIC self-budgets
#SBATCH --time=04:00:00
#
# One independent MOSAIC case per array task. This is the cluster shape of
# scripts/run_catio3_hkl40_pipelined.py: cases are scientifically independent
# and resume from durable state on any node. NOTE the stage-1 payload store
# is scoped by the run's scientific digest, which includes the MASK — the
# four cases differ exactly in their mask equations, so each array task
# keeps its OWN store subtree (~75 GB each at hkl40 scale, 4x total, not one
# shared copy). Point MOSAIC_STREAMING_PAYLOAD_STORE at a parallel FS with
# room for all four; only reruns of the SAME case reuse a store.
#
# Nothing below is machine-specific:
#   - runtime.dask.max_workers is "auto" -> one worker per GPU SLURM grants
#   - all budgets (VRAM pool, RAM admission, tile sizes) probe the node
#   - checkpoint identity is worker-count invariant: a job preempted on a
#     4-GPU node resumes on an 8-GPU node without recomputation (streaming
#     slot count and partition capacity are plan constants in the parameter
#     digest, never live worker counts)
#
# For MULTI-NODE-PER-CASE runs, switch the config's runtime.dask.backend to
# "slurm" (dask-jobqueue) or "mpi" (dask-mpi); the store must then be on the
# shared FS.

set -euo pipefail

ROOT="${MOSAIC_ROOT:-$SLURM_SUBMIT_DIR}"
CASES=(all sphere rod rest)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"

source "$ROOT/.venv/bin/activate"
SITE="$ROOT/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Scratch on node-local storage; the payload store on the shared filesystem
# so every node (and every restart) reuses stage-1.
NODE_SCRATCH="${SLURM_TMPDIR:-/tmp/$USER/mosaic-$SLURM_JOB_ID}"
mkdir -p "$NODE_SCRATCH"
export DASK_LOCAL_DIR="$NODE_SCRATCH/dask"
export MOSAIC_RESIDUAL_LATTICE_SCRATCH="$NODE_SCRATCH/lattice"
export MOSAIC_WORKER_SCRATCH_ROOT="$NODE_SCRATCH/worker"
export MOSAIC_RESIDUAL_SHARD_SCRATCH_ROOT="$NODE_SCRATCH/shard"
export MOSAIC_STREAMING_PAYLOAD_STORE="${MOSAIC_STREAMING_PAYLOAD_STORE:-$ROOT/stage1_store}"

export MOSAIC_NUFFT_EXECUTION_POLICY=gpu-required
export MOSAIC_SCATTERING_STAGE2_STREAMING=1
export MOSAIC_GPU_ALLOW_MULTI_THREAD_WORKER=1
export MOSAIC_NUFFT_SLOTS_PER_WORKER=2
export MOSAIC_DASK_GPU_RESOURCE=2
export MOSAIC_RESIDUAL_PREFETCH_FACTOR=8
export MOSAIC_RESIDUAL_LATTICE_RAM_FRACTION=0.2
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=$((64<<20))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec python -m core.main \
  "$ROOT/examples/config_3D/displacement/run_parameters_hkl40_${CASE}.json"
