#!/usr/bin/env bash
set -Eeuo pipefail

# MOSAIC CUDA install for Linux/WSL without Conda.
#
# Default path:
#   - install basic apt prerequisites when missing
#   - install uv when missing
#   - install NVIDIA CUDA 12.4 toolkit/runtime when CUDA 12 libs are missing
#   - create .venv with uv-managed Python 3.11
#   - install MOSAIC editable with the cuda12 extra
#   - write .mosaic_cuda_env and wire it into .venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
CUDA_MAJOR_MINOR="${CUDA_MAJOR_MINOR:-12.4}"
CUDA_APT_PACKAGE="${CUDA_APT_PACKAGE:-cuda-toolkit-12-4}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/uv}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
MOSAIC_REQUIRE_GPU="${MOSAIC_REQUIRE_GPU:-0}"

INSTALL_APT=1
INSTALL_CUDA_APT=1
RUN_SMOKE=0
GPU_CHECK=1
INSTALL_HPC_EXTRAS=0

usage() {
    cat <<'EOF'
Usage: ./setup_mosaic_uv_cuda.sh [options]

Installs MOSAIC CUDA dependencies without Conda, intended for a clean WSL
Ubuntu environment with the Windows NVIDIA driver already installed.

Options:
  --python VERSION       Python version for uv to install/use (default: 3.11)
  --venv PATH            Virtual environment path (default: .venv)
  --no-apt               Do not install apt prerequisites or CUDA toolkit
  --no-cuda-apt          Do not install NVIDIA CUDA packages through apt
  --no-gpu-check         Skip CUDA device visibility check
  --hpc-extras           Also install dask-jobqueue and dask-mpi
  --smoke                Run the small CPU-only MOSAIC smoke example
  -h, --help             Show this help

Environment overrides:
  PYTHON_VERSION=3.12
  VENV_DIR=.venv
  UV_CACHE_DIR=/tmp/uv-cache
  CUDA_MAJOR_MINOR=12.4
  CUDA_APT_PACKAGE=cuda-toolkit-12-4
  MOSAIC_CUDA_HOME=/usr/local/cuda-12.4
  MOSAIC_REQUIRE_GPU=1

After setup:
  source .venv/bin/activate
  mosaic examples/config_1D/displacement/run_parameters.json

GPU example:
  CUDA_VISIBLE_DEVICES=0 mosaic examples/config_2D/chemical_ordering/run_parameters.json
EOF
}

log() {
    printf '\n>>> %s\n' "$*"
}

warn() {
    printf '\nWARNING: %s\n' "$*" >&2
}

die() {
    printf '\nERROR: %s\n' "$*" >&2
    exit 1
}

run_sudo() {
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    else
        command -v sudo >/dev/null 2>&1 || die "sudo is required for apt installation. Re-run with --no-apt to skip system package installation."
        sudo "$@"
    fi
}

is_wsl() {
    [[ -n "${WSL_DISTRO_NAME:-}" ]] || grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null
}

shell_quote() {
    local value="$1"
    printf "'%s'" "${value//\'/\'\\\'\'}"
}

abs_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$SCRIPT_DIR" "$path"
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --python)
                [[ $# -ge 2 ]] || die "--python requires a version"
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --venv)
                [[ $# -ge 2 ]] || die "--venv requires a path"
                VENV_DIR="$2"
                shift 2
                ;;
            --no-apt)
                INSTALL_APT=0
                INSTALL_CUDA_APT=0
                shift
                ;;
            --no-cuda-apt|--skip-cuda-apt)
                INSTALL_CUDA_APT=0
                shift
                ;;
            --no-gpu-check)
                GPU_CHECK=0
                shift
                ;;
            --hpc-extras)
                INSTALL_HPC_EXTRAS=1
                shift
                ;;
            --smoke)
                RUN_SMOKE=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
    done
}

ensure_uv_cache() {
    if ! mkdir -p "$UV_CACHE_DIR" 2>/dev/null || [[ ! -w "$UV_CACHE_DIR" ]]; then
        warn "UV_CACHE_DIR=$UV_CACHE_DIR is not writable; using $SCRIPT_DIR/.uv-cache"
        UV_CACHE_DIR="$SCRIPT_DIR/.uv-cache"
        mkdir -p "$UV_CACHE_DIR"
    fi
    export UV_CACHE_DIR UV_LINK_MODE
}

ensure_apt_prereqs() {
    [[ "$INSTALL_APT" -eq 1 ]] || return 0
    command -v apt-get >/dev/null 2>&1 || die "apt-get was not found. Re-run with --no-apt and install prerequisites manually."

    local packages=(
        ca-certificates
        curl
        gnupg
        lsb-release
        build-essential
        pkg-config
        libgomp1
    )
    local missing=()
    local pkg

    for pkg in "${packages[@]}"; do
        dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done

    if [[ "${#missing[@]}" -gt 0 ]]; then
        log "Installing apt prerequisites: ${missing[*]}"
        run_sudo apt-get update
        run_sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
    else
        log "Apt prerequisites are already installed"
    fi
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        log "Using uv at $UV_BIN"
        return 0
    fi

    command -v curl >/dev/null 2>&1 || die "curl is required to install uv"
    log "Installing uv with the official Astral installer"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    curl -fsSL https://astral.sh/uv/install.sh -o "$tmp_dir/uv-install.sh"
    sh "$tmp_dir/uv-install.sh"
    export PATH="$HOME/.local/bin:$PATH"

    command -v uv >/dev/null 2>&1 || die "uv installation finished, but uv is still not on PATH. Add $HOME/.local/bin to PATH and rerun."
    UV_BIN="$(command -v uv)"
}

cuda_lib_dir_for_home() {
    local cuda_home="$1"
    local candidate
    for candidate in \
        "$cuda_home/targets/x86_64-linux/lib" \
        "$cuda_home/lib64" \
        "$cuda_home/lib"; do
        if [[ -e "$candidate/libcudart.so.12" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

find_cuda_home() {
    local candidates=()
    local path

    if [[ -n "${MOSAIC_CUDA_HOME:-}" ]]; then
        candidates+=("$MOSAIC_CUDA_HOME")
    fi

    candidates+=(
        "/usr/local/cuda-${CUDA_MAJOR_MINOR}"
        "/usr/local/cuda-12.4"
        "/usr/local/cuda-12.3"
        "/usr/local/cuda-12.2"
        "/usr/local/cuda-12.1"
        "/usr/local/cuda-12.0"
        "/usr/local/cuda"
    )

    while IFS= read -r path; do
        candidates+=("$path")
    done < <(find /usr/local -maxdepth 1 -type d -name 'cuda-12*' 2>/dev/null | sort -Vr || true)

    local seen=":"
    for path in "${candidates[@]}"; do
        [[ -d "$path" ]] || continue
        case "$seen" in
            *":$path:"*) continue ;;
        esac
        seen="${seen}${path}:"
        if cuda_lib_dir_for_home "$path" >/dev/null 2>&1; then
            printf '%s\n' "$path"
            return 0
        fi
    done

    return 1
}

cuda_repo_id() {
    if is_wsl; then
        printf 'wsl-ubuntu\n'
        return 0
    fi

    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        if [[ -n "${VERSION_ID:-}" ]]; then
            printf 'ubuntu%s\n' "${VERSION_ID//./}"
            return 0
        fi
    fi

    die "Could not determine Ubuntu release for CUDA apt repo"
}

install_cuda_toolkit() {
    if find_cuda_home >/dev/null 2>&1; then
        log "CUDA 12 runtime libraries are already available"
        return 0
    fi

    [[ "$INSTALL_CUDA_APT" -eq 1 ]] || die "CUDA 12 runtime libraries were not found. Install CUDA 12.4 or set MOSAIC_CUDA_HOME, then rerun."
    command -v apt-get >/dev/null 2>&1 || die "apt-get was not found, and CUDA 12 runtime libraries are missing"
    command -v curl >/dev/null 2>&1 || die "curl is required to install the NVIDIA CUDA apt keyring"
    command -v dpkg >/dev/null 2>&1 || die "dpkg is required to install the NVIDIA CUDA apt keyring"

    if is_wsl; then
        log "Installing CUDA toolkit from NVIDIA's WSL apt repository"
    else
        warn "This does not look like WSL. The script will use the Ubuntu CUDA repo and install $CUDA_APT_PACKAGE, not a display driver."
    fi

    local repo_id keyring_url tmp_dir
    repo_id="$(cuda_repo_id)"
    keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/cuda-keyring_1.1-1_all.deb"
    tmp_dir="$(mktemp -d)"

    log "Installing NVIDIA CUDA apt keyring for ${repo_id}"
    curl -fsSL "$keyring_url" -o "$tmp_dir/cuda-keyring.deb"
    run_sudo dpkg -i "$tmp_dir/cuda-keyring.deb"
    run_sudo apt-get update

    log "Installing $CUDA_APT_PACKAGE"
    run_sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y "$CUDA_APT_PACKAGE"

    find_cuda_home >/dev/null 2>&1 || die "Installed $CUDA_APT_PACKAGE, but CUDA 12 runtime libraries are still not visible under /usr/local/cuda-12*"
}

install_python_env() {
    local venv_path="$1"

    log "Installing Python $PYTHON_VERSION with uv if needed"
    "$UV_BIN" python install "$PYTHON_VERSION"

    log "Creating/updating virtual environment at $venv_path"
    "$UV_BIN" venv --python "$PYTHON_VERSION" "$venv_path"

    log "Installing MOSAIC editable with CUDA 12 Python dependencies"
    "$UV_BIN" pip install --python "$venv_path/bin/python" -e '.[cuda12]'

    if [[ "$INSTALL_HPC_EXTRAS" -eq 1 ]]; then
        log "Installing optional HPC Dask extras"
        "$UV_BIN" pip install --python "$venv_path/bin/python" dask-jobqueue dask-mpi
    fi
}

write_cuda_env() {
    local cuda_home="$1"
    local cuda_lib_dir="$2"
    local env_file="$SCRIPT_DIR/.mosaic_cuda_env"
    local q_cuda_home q_cuda_lib_dir q_wsl_lib

    q_cuda_home="$(shell_quote "$cuda_home")"
    q_cuda_lib_dir="$(shell_quote "$cuda_lib_dir")"
    q_wsl_lib="$(shell_quote "/usr/lib/wsl/lib")"

    log "Writing CUDA runtime environment to $env_file"
    cat > "$env_file" <<EOF
# Generated by setup_mosaic_uv_cuda.sh. Safe to source from bash.
export MOSAIC_CUDA_HOME=${q_cuda_home}
export MOSAIC_CUDA_LIB_DIR=${q_cuda_lib_dir}

_mosaic_prepend_ld_library_path() {
    [ -d "\$1" ] || return 0
    case ":\${LD_LIBRARY_PATH:-}:" in
        *":\$1:"*) ;;
        *) export LD_LIBRARY_PATH="\$1\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" ;;
    esac
}

_mosaic_prepend_ld_library_path ${q_wsl_lib}
_mosaic_prepend_ld_library_path ${q_cuda_lib_dir}
unset -f _mosaic_prepend_ld_library_path
EOF
}

patch_venv_activate() {
    local venv_path="$1"
    local activate_file="$venv_path/bin/activate"
    local env_file="$SCRIPT_DIR/.mosaic_cuda_env"
    local q_env_file

    [[ -f "$activate_file" ]] || return 0
    if grep -q 'MOSAIC CUDA env' "$activate_file"; then
        return 0
    fi

    q_env_file="$(shell_quote "$env_file")"
    log "Wiring CUDA environment into $activate_file"
    cat >> "$activate_file" <<EOF

# >>> MOSAIC CUDA env >>>
if [ -f ${q_env_file} ]; then
    . ${q_env_file}
fi
# <<< MOSAIC CUDA env <<<
EOF
}

write_wrappers() {
    local venv_path="$1"
    local env_file="$SCRIPT_DIR/.mosaic_cuda_env"
    local wrapper="$venv_path/bin/mosaic-cuda"
    local q_env_file q_mosaic

    q_env_file="$(shell_quote "$env_file")"
    q_mosaic="$(shell_quote "$venv_path/bin/mosaic")"

    log "Writing convenience wrapper $wrapper"
    cat > "$wrapper" <<EOF
#!/usr/bin/env bash
set -euo pipefail
. ${q_env_file}
exec ${q_mosaic} "\$@"
EOF
    chmod +x "$wrapper"
}

check_nvidia_smi() {
    [[ "$GPU_CHECK" -eq 1 ]] || return 0

    local smi=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        smi="$(command -v nvidia-smi)"
    elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
        smi="/usr/lib/wsl/lib/nvidia-smi"
    fi

    if [[ -z "$smi" ]]; then
        warn "nvidia-smi was not found. In WSL, install/update the Windows NVIDIA driver before expecting GPU execution."
        return 0
    fi

    if ! "$smi" >/dev/null 2>&1; then
        warn "nvidia-smi exists but cannot access the GPU. CUDA packages can be installed, but GPU execution will fail until WSL GPU access works."
    else
        log "nvidia-smi can see the GPU"
    fi
}

sanity_check_python_cuda() {
    local venv_path="$1"
    local env_file="$SCRIPT_DIR/.mosaic_cuda_env"

    log "Checking Python CUDA imports"
    # shellcheck disable=SC1090
    . "$env_file"

    set +e
    "$venv_path/bin/python" - <<'PY'
import sys

try:
    import cupy as cp
    import cufinufft
    import dask_cuda
    import finufft
except Exception as exc:
    print(f"IMPORT_ERROR: {type(exc).__name__}: {exc}")
    sys.exit(2)

print(f"python={sys.version.split()[0]}")
print(f"cupy={cp.__version__}")
print(f"cufinufft={cufinufft.__version__}")
print(f"finufft={finufft.__version__}")
print(f"dask_cuda={getattr(dask_cuda, '__version__', 'unknown')}")

try:
    print(f"cuda_device_count={cp.cuda.runtime.getDeviceCount()}")
except Exception as exc:
    print(f"CUDA_DEVICE_WARNING: {type(exc).__name__}: {exc}")
    sys.exit(20)
PY
    local status=$?
    set -e

    case "$status" in
        0)
            log "CUDA Python import and device checks passed"
            ;;
        20)
            if [[ "$MOSAIC_REQUIRE_GPU" == "1" ]]; then
                die "CUDA Python imports worked, but no CUDA device was usable"
            fi
            warn "CUDA Python imports worked, but no CUDA device was usable. Fix WSL/driver GPU access before running CUDA examples."
            ;;
        *)
            die "CUDA Python import check failed"
            ;;
    esac
}

run_smoke_example() {
    local venv_path="$1"
    [[ "$RUN_SMOKE" -eq 1 ]] || return 0

    log "Running CPU-only MOSAIC smoke example"
    MOSAIC_NUFFT_CPU_ONLY=1 "$venv_path/bin/python" scripts/smoke_example.py
}

main() {
    parse_args "$@"

    local venv_path cuda_home cuda_lib_dir
    venv_path="$(abs_path "$VENV_DIR")"

    log "MOSAIC uv CUDA setup"
    if is_wsl; then
        log "WSL detected"
    else
        warn "WSL was not detected. This script can still work on Ubuntu, but it is primarily tuned for WSL."
    fi

    ensure_uv_cache
    ensure_apt_prereqs
    ensure_uv
    install_cuda_toolkit

    cuda_home="$(find_cuda_home)" || die "CUDA 12 runtime libraries were not found"
    cuda_lib_dir="$(cuda_lib_dir_for_home "$cuda_home")" || die "Could not find CUDA runtime library directory for $cuda_home"

    log "Using CUDA home: $cuda_home"
    log "Using CUDA lib dir: $cuda_lib_dir"

    install_python_env "$venv_path"
    write_cuda_env "$cuda_home" "$cuda_lib_dir"
    patch_venv_activate "$venv_path"
    write_wrappers "$venv_path"
    check_nvidia_smi
    sanity_check_python_cuda "$venv_path"
    run_smoke_example "$venv_path"

    cat <<EOF

>>> Setup complete

Use MOSAIC with:
  source ${VENV_DIR}/bin/activate
  mosaic examples/config_1D/displacement/run_parameters.json

CUDA wrapper:
  ${VENV_DIR}/bin/mosaic-cuda examples/config_2D/chemical_ordering/run_parameters.json

The CUDA library path is stored in:
  .mosaic_cuda_env
EOF
}

main "$@"
