#!/usr/bin/env bash
#SBATCH --job-name=mosaic_driver
#SBATCH --output=%x.o.%j
#SBATCH --error=%x.e.%j
#SBATCH --time=04:10:00          # slightly longer than workers
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0             # scheduler needs no GPU
#SBATCH --partition=gpu          # adjust to your cluster
#SBATCH --export=ALL             # pass our exports to the job

set -euo pipefail

# ── Cluster configuration via ENV vars ────────────────────────────
export DASK_BACKEND=slurm

# total number of worker jobs and their shape
export DASK_MAX_WORKERS=32
export DASK_THREADS_PER_WORKER=4
export GPUS_PER_JOB=1            # each worker gets 1 GPU

# job‑queue defaults (per worker)
export DASK_WALLTIME="04:00:00"  # matches --time above
export DASK_MEMORY="32GB"
export DASK_LOCAL_DIR="/scratch/${USER}/dask"

# SLURM‑specific extra directives automatically used by helper
export DASK_SLURM_EXTRA="--account=myproj --partition=gpu"

# Log directory for worker .o/.e files
export MOSAIC_LOG_DIR="$HOME/mosaic_logs"

# Optional: hide per‑worker dashboards
export DASK_WORKER_DASHBOARD=0

# Python for scheduler & workers
export DASK_PYTHON="$HOME/venvs/mosaic/bin/python"

# Activate environment and launch MOSAIC entry point
source "$HOME/venvs/mosaic/bin/activate"
"$DASK_PYTHON" /path/to/main.py "$@"