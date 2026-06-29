import json
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


def test_cleanup_cli_invokes_cleanup_report(monkeypatch, tmp_path, capsys):
    from core.entrypoints import main as entrypoint

    calls = []

    class Report:
        removed_paths = ("removed.tmp",)
        retained_paths = ("kept.tmp",)
        skipped_reasons = ("active public run is retained",)

    def fake_cleanup(**kwargs):
        calls.append(kwargs)
        return Report()

    monkeypatch.setattr("core.storage.cleanup.cleanup_run_artifacts", fake_cleanup)

    result = entrypoint.cli(
        [
            "cleanup",
            "--output-dir",
            str(tmp_path),
            "--run-digest",
            "run123",
            "--temp-file-grace-seconds",
            "5",
            "--remove-superseded-run",
            "--json",
        ]
    )

    assert result == 0
    assert calls == [
        {
            "output_dir": str(tmp_path),
            "run_digest": "run123",
            "temp_file_grace_seconds": 5.0,
            "remove_superseded_run": True,
        }
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed_paths"] == ["removed.tmp"]
    assert payload["retained_paths"] == ["kept.tmp"]
    assert payload["skipped_reasons"] == ["active public run is retained"]
