"""E2E gate for the amorphous SiO2 imposed-displacement case.

Runs the pipeline on DisplacedGlass.data against the FinalGlass.data
reference (config_3D/amorphous_sio2), then compares the decoded per-site
displacements with the imposed field u_true.npy under the box minimum
image.

Two metrics, two roles:
- ACCURACY vs the imposed truth — physics-limited (band limit, patch
  discretization, linear decoder), reported always, gated loosely
  (--max-err, default 0.05 A: order of the imposed field itself, i.e.
  "the decode is real, not noise").
- REPRODUCIBILITY — with --reference <csv-dir>, decoded output is
  compared against a previous run at the crystal pipeline's run-vs-run
  gate (1e-10 A).

Usage:
  python scripts/amorphous_gate.py             # run + accuracy gate
  python scripts/amorphous_gate.py --skip-run  # gate existing output
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CASE_DIR = REPO_ROOT / "examples" / "config_3D" / "amorphous_sio2"
OUTPUT_DIR = CASE_DIR / "output_amorphous_small"
BOX_L = 35.339987063975235  # FinalGlass box edge, Angstrom


def run_case() -> None:
    sys.path.insert(0, str(REPO_ROOT / "examples"))
    from example_runner import mosaic_environment

    env = mosaic_environment(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-m", "core.main", "run_parameters_small.json"],
        cwd=CASE_DIR,
        env=env,
    )
    if completed.returncode != 0:
        raise SystemExit(f"pipeline run failed: rc={completed.returncode}")


def load_decoded(output_dir: Path) -> pd.DataFrame:
    proc_dir = output_dir / "processed_point_data"
    csv_paths = sorted(proc_dir.glob("chunk_*_site_displacements.csv"))
    if not csv_paths:
        raise SystemExit(f"no displacement CSVs under {proc_dir}")
    frames = [pd.read_csv(path, sep="\t") for path in csv_paths]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("central_point_id")
        .reset_index(drop=True)
    )


def minimum_image(delta: np.ndarray) -> np.ndarray:
    return delta - BOX_L * np.rint(delta / BOX_L)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--max-err", type=float, default=0.05)
    parser.add_argument("--reference", type=Path, default=None,
                        help="previous run's processed_point_data dir for the 1e-10 reproducibility gate")
    args = parser.parse_args()

    if not args.skip_run:
        run_case()

    decoded = load_decoded(OUTPUT_DIR)
    u_true = np.load(CASE_DIR / "u_true.npy")

    ids = decoded["central_point_id"].to_numpy()
    u_hat = decoded[["ux", "uy", "uz"]].to_numpy(dtype=float)
    truth = u_true[ids]
    err = minimum_image(u_hat - truth)
    err_mag = np.linalg.norm(err, axis=1)

    print(f"sites decoded      : {len(decoded)} / {len(u_true)}")
    print(f"imposed  RMS |u|   : {np.sqrt((truth**2).sum(1).mean()):.6f} A")
    print(f"decoded  RMS |u|   : {np.sqrt((u_hat**2).sum(1).mean()):.6f} A")
    print(f"accuracy max |du|  : {err_mag.max():.3e} A")
    print(f"accuracy RMS |du|  : {np.sqrt((err_mag**2).mean()):.3e} A")

    failed = False
    if len(decoded) != len(u_true):
        print(f"FAIL: {len(u_true) - len(decoded)} sites missing")
        failed = True
    if err_mag.max() > args.max_err:
        print(f"FAIL: accuracy max |du| {err_mag.max():.3e} > {args.max_err}")
        failed = True

    if args.reference is not None:
        ref_frames = [
            pd.read_csv(path, sep="\t")
            for path in sorted(args.reference.glob("chunk_*_site_displacements.csv"))
        ]
        reference = (
            pd.concat(ref_frames, ignore_index=True)
            .sort_values("central_point_id")
            .reset_index(drop=True)
        )
        repro = np.abs(
            decoded[["ux", "uy", "uz"]].to_numpy() - reference[["ux", "uy", "uz"]].to_numpy()
        ).max()
        print(f"reproducibility max|diff| : {repro:.3e} A (gate 1e-10)")
        if repro >= 1e-10:
            print("FAIL: reproducibility gate")
            failed = True

    print("GATE:", "FAIL" if failed else "PASS")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
