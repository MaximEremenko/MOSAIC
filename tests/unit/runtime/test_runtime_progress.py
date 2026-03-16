from __future__ import annotations

import builtins
import importlib
import sys


def test_progress_bar_can_be_forced_without_tty(monkeypatch) -> None:
    import core.runtime.progress as progress

    monkeypatch.setenv("MOSAIC_FORCE_PROGRESS", "1")
    monkeypatch.setattr(progress.sys.stderr, "isatty", lambda: False)

    bar = progress.progress_bar(1, desc="forced", unit="items")
    try:
        assert getattr(bar, "disable", False) is False
    finally:
        bar.close()


def test_progress_overrides_can_be_configured_without_env(monkeypatch) -> None:
    import core.runtime.progress as progress

    monkeypatch.delenv("MOSAIC_FORCE_PROGRESS", raising=False)
    monkeypatch.delenv("MOSAIC_TASK_PROGRESS", raising=False)
    try:
        progress.configure_progress(force_progress=True, task_progress=True)
        monkeypatch.setattr(progress.sys.stderr, "isatty", lambda: False)
        bar = progress.progress_bar(1, desc="configured", unit="items")
        try:
            assert getattr(bar, "disable", False) is False
        finally:
            bar.close()
        assert progress.task_progress_enabled(False) is True
    finally:
        progress.configure_progress(force_progress=None, task_progress=None)


def test_progress_module_falls_back_when_tqdm_missing(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tqdm" or name.startswith("tqdm."):
            raise ModuleNotFoundError("No module named 'tqdm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("core.runtime.progress", None)
    try:
        progress = importlib.import_module("core.runtime.progress")

        with progress.logging_redirect_tqdm():
            with progress.tqdm(total=1, desc="noop", unit="items") as pbar:
                pbar.update(1)
                pbar.refresh()

        assert hasattr(pbar, "close")
    finally:
        sys.modules.pop("core.runtime.progress", None)
        importlib.import_module("core.runtime.progress")
