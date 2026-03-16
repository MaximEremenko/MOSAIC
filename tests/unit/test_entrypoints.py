import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_core_main_direct_script_bootstrap(tmp_path):
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; runpy.run_path('main.py', run_name='script_import')",
        ],
        cwd=ROOT / "core",
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_core_main_package_entry_bootstrap(tmp_path):
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; runpy.run_module('core.main', run_name='package_import')",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
