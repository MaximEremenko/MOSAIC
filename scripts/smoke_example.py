from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_FILE = ROOT / "examples" / "config_1D" / "displacement" / "run_parameters.json"
OUTPUT_DIR = ROOT / "examples" / "config_1D" / "displacement" / "output_displacement"


def main() -> int:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    env = dict(os.environ)
    env["MOSAIC_NUFFT_CPU_ONLY"] = "1"
    subprocess.run(
        [sys.executable, "-m", "core.main", str(RUN_FILE)],
        cwd=ROOT,
        env=env,
        check=True,
        timeout=180,
    )

    processed = OUTPUT_DIR / "processed_point_data"
    required = [
        processed / "point_data.hdf5",
        processed / "point_reciprocal_space_associations.db",
        processed / "point_reciprocal_space_data.hdf5",
        processed / "point_data_chunk_0_amplitudes.hdf5",
        processed / "point_data_chunk_0_amplitudes_av.hdf5",
        processed / "point_data_chunk_0_shapeNd.hdf5",
        processed / "point_data_chunk_0_applied_interval_ids.hdf5",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Canonical smoke example did not produce expected artifacts:\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    print(f"Smoke example completed: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
