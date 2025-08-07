#!/usr/bin/env bash
#$ -N mosaic_driver
#$ -cwd -V
#$ -l gpu=0          # scheduler needs no GPU
#$ -pe mpi 1
#$ -l h_rt=04:10:00
#$ -o $HOME/mosaic_logs/sched.o.$JOB_ID
#$ -e $HOME/mosaic_logs/sched.e.$JOB_ID

set -euo pipefail

# ── ALL CONFIG VIA ENV ───────────────────────────
export DASK_BACKEND=sge
export DASK_MAX_WORKERS=16
export DASK_THREADS_PER_WORKER=4
export GPUS_PER_JOB=1
export DASK_WALLTIME="04:00:00"
export DASK_MEMORY="32GB"
export DASK_LOCAL_DIR="/scratch/${USER}/dask"

# SGE‑specific extras picked up by _build_job_extra()
export DASK_PE="-pe gpu 1"
export DASK_GPU="-l gpu=1"
export DASK_HOST="-l hostname=!login*"

export MOSAIC_LOG_DIR="$HOME/mosaic_logs"
export DASK_PYTHON="$HOME/venvs/mosaic/bin/python"
export DASK_WORKER_DASHBOARD=0        # hide per‑worker UI

# Activate venv and launch your real entry point
source "$HOME/venvs/mosaic/bin/activate"
"$DASK_PYTHON" /path/to/main.py "$@"
