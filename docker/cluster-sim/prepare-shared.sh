#!/bin/bash
# Stage the small displacement cases onto the simulated shared filesystem
# (run on the HOST after up.sh). One shared config directory for all four
# cases — matching the repo layout — because sphere/rod/rest resolve the
# decoder trained by the 'all' case through a relative cache_path
# ('./output_disp_small_all/...'). Configs, structure files, outputs, and
# the stage-1 payload store all live on NFS — the cluster layout.
set -euo pipefail
cd "$(dirname "$0")"

REPO="$(cd ../.. && pwd)"
SIM_ROOT="${SIM_ROOT:-$HOME/mosaic-sim}"
SRC="$REPO/examples/config_3D/displacement"
DEST="$SIM_ROOT/shared/configs/displacement"

mkdir -p "$DEST"
cp "$SRC/Catio3_small.rmc6f" "$SRC/Catio3_small_average.rmc6f" "$DEST/"
for case in all sphere rod rest; do
    python3 - "$SRC" "$DEST" "$case" <<'EOF'
import json, sys
src, dest, case = sys.argv[1:4]
cfg = json.load(open(f"{src}/input_parameters_small_{case}.json"))
cfg["paths"]["config_root"] = "."
cfg["paths"]["output_directory"] = f"./output_disp_small_{case}"
cfg["processing"]["fresh_start"] = False
dask = cfg["runtime"]["dask"]
dask["max_workers"] = "auto"      # one worker per GPU the node grants
dask["threads_per_worker"] = 2
json.dump(cfg, open(f"{dest}/input_parameters_small_{case}.json", "w"), indent=2)
json.dump({"input_parameters_path": f"./input_parameters_small_{case}.json"},
          open(f"{dest}/run_parameters_small_{case}.json", "w"), indent=2)
EOF
done
mkdir -p "$SIM_ROOT/shared/stage1_store"
echo "staged: $DEST/run_parameters_small_{all,sphere,rod,rest}.json"
