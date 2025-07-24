#!/usr/bin/env bash
# Run MOSAIC locally using all visible GPUs via dask‑cuda

set -euo pipefail

# ── Local GPU configuration ───────────────────────────────────────
export DASK_BACKEND=cuda-local      # tells helper to use LocalCUDACluster
export DASK_MAX_WORKERS=auto        # one worker per GPU (dask‑cuda default)
export DASK_THREADS_PER_WORKER=4    # CPU threads per worker
export DASK_LOCAL_DIR="/tmp/dask"   # fast NVMe scratch if you have it
export DASK_WORKER_DASHBOARD=0      # quieter console

# (Optional) Restrict to specific GPUs, e.g. “0,2,3”
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Where to save logs if you want them
export MOSAIC_LOG_DIR="$HOME/mosaic_logs"

# Python interpreter
export DASK_PYTHON="$HOME/venvs/mosaic/bin/python"

# Activate venv and run MOSAIC
source "$HOME/venvs/mosaic/bin/activate"
"$DASK_PYTHON" /path/to/main.py "$@"
