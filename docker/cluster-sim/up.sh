#!/bin/bash
# Bring up the simulated cluster. Requires: docker + compose v2 + NVIDIA
# container toolkit, and the nfs/nfsd kernel modules on the host.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd ../.. && pwd)"
SIM_ROOT="${SIM_ROOT:-$HOME/mosaic-sim}"
mkdir -p "$SIM_ROOT/shared" "$SIM_ROOT"/scratch/node{1,2,3}

sudo modprobe nfs nfsd

cat > .env <<EOF
REPO=$REPO
SIM_ROOT=$SIM_ROOT
EOF

sudo docker compose build
sudo docker compose up -d --wait
sudo docker compose ps
