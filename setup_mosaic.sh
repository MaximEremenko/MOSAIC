#!/usr/bin/env bash
set -euo pipefail  # Exit on error, unset vars, failed pipes

# ====== CONFIGURABLE SECTION ======
ENV_NAME="mosaic"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.4.0"
FINUFFT_VERSION="2.2.0"
CUFINUFFT_VERSION="2.5.1"
DASK_CUDA_MIN_VERSION="26.4"
# ================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ">>> Writing environment.yml ..."
cat <<EOF > environment.yml
name: $ENV_NAME
channels:
  - conda-forge
  - nvidia/label/cuda-${CUDA_VERSION}
dependencies:
  - python=${PYTHON_VERSION}
  - numpy
  - scipy
  - pandas
  - matplotlib
  - h5py
  - dask=2026.1.1
  - distributed=2026.1.1
  - dask-jobqueue
  - dask-mpi
  - numba
  - pyyaml
  - requests
  - sympy
  - tqdm
  - fftw
  - cuda-toolkit=${CUDA_VERSION}
  - pip
  - pip:
      - cuda-bindings==12.*
      - finufft==${FINUFFT_VERSION}
      - cupy-cuda12x
      - cufinufft==${CUFINUFFT_VERSION}
      - dask-cuda>=${DASK_CUDA_MIN_VERSION}
EOF

echo ">>> Creating or updating conda environment '$ENV_NAME' ..."
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "    Environment exists, updating ..."
    conda env update -n "$ENV_NAME" -f environment.yml
else
    conda env create -f environment.yml
fi

echo ">>> Activating environment ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ">>> Installing MOSAIC (editable) ..."
pip install -e .

echo ">>> Ensuring CUDA wheel stack ..."
python -m pip install --upgrade \
    "cuda-bindings==12.*" \
    "finufft==${FINUFFT_VERSION}" \
    "cupy-cuda12x" \
    "cufinufft==${CUFINUFFT_VERSION}" \
    "dask-cuda>=${DASK_CUDA_MIN_VERSION}"

echo ">>> Installing CUDA library path activation hook ..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/mosaic_cuda_libs.sh" <<EOF
export MOSAIC_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}"
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/mosaic_cuda_libs.sh" <<'EOF'
if [ -n "${MOSAIC_OLD_LD_LIBRARY_PATH+x}" ]; then
    export LD_LIBRARY_PATH="$MOSAIC_OLD_LD_LIBRARY_PATH"
    unset MOSAIC_OLD_LD_LIBRARY_PATH
fi
EOF
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo ">>> CUDA package sanity check ..."
python - <<'PY'
import cufinufft
import cupy as cp

print(f"cufinufft={cufinufft.__version__}")
print(f"cupy={cp.__version__}")
try:
    print(f"cuda_device_count={cp.cuda.runtime.getDeviceCount()}")
except Exception as exc:
    print(f"WARNING: CUDA runtime check failed on this node: {type(exc).__name__}: {exc}")
PY

echo ">>> Setup complete!"
echo "To activate your environment in the future: conda activate $ENV_NAME"
