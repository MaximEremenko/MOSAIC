from __future__ import annotations

import builtins
import importlib
import sys

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
