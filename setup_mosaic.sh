#!/usr/bin/env bash
set -e  # Exit on error

# ====== CONFIGURABLE SECTION ======
ENV_NAME="mosaic"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.4.0"
# ================================

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
  - dask
  - dask_cuda
  - dask_jobqueue
  - dask_mpi
  - cupy
  - numba
  - requests
  - sympy
  - fftw
  - cuda-toolkit
  - pip
EOF

echo ">>> Creating conda environment '$ENV_NAME' ..."
conda env create -f environment.yml || {
    echo "Environment already exists, updating it ..."
    conda env update -f environment.yml
}

echo ">>> Activating environment ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ">>> Cloning FINUFFT repository ..."
git clone --depth 1 https://github.com/flatironinstitute/finufft.git || {
    echo "Repository already cloned. Skipping."
}

cd finufft

echo ">>> Installing FINUFFT and cuFINUFFT from source (no binaries)..."
pip install --no-binary finufft python/finufft
pip install --no-binary cufinufft python/cufinufft

cd ..

echo ">>> Setup complete!"
echo "To activate your environment in the future: conda activate $ENV_NAME"
