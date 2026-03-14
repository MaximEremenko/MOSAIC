from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

try:
    from tqdm import tqdm as _tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm as _logging_redirect_tqdm
except ModuleNotFoundError:
    class _NoopTqdm:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self) -> "_NoopTqdm":
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def update(self, n: int = 1) -> None:
            return None

        def refresh(self) -> None:
            return None

        def close(self) -> None:
            return None

    def _tqdm(*args, **kwargs) -> _NoopTqdm:
        return _NoopTqdm()

    @contextmanager
    def _logging_redirect_tqdm():
        yield


tqdm = _tqdm
logging_redirect_tqdm = _logging_redirect_tqdm

TIMER = time.perf_counter


@contextmanager
def timed(label: str, *, logger: logging.Logger | None = None):
    active_logger = logger or logging.getLogger(__name__)
    t0 = TIMER()
    try:
        yield
    finally:
        active_logger.info("%s took %.3f s", label, TIMER() - t0)


@contextmanager
def quiet_loggers(*names: str):
    logs = [logging.getLogger(name) for name in names]
    previous_levels = [log.level for log in logs]
    try:
        for log in logs:
            log.setLevel(max(logging.WARNING, log.level))
        yield
    finally:
        for log, level in zip(logs, previous_levels):
            log.setLevel(level)


def progress_bar(total: int, *, desc: str, unit: str):
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        smoothing=0,
        miniters=1,
        mininterval=0.1,
        leave=True,
        disable=(total <= 0 or not sys.stderr.isatty()),
    )

__all__ = [
    "TIMER",
    "logging_redirect_tqdm",
    "progress_bar",
    "quiet_loggers",
    "timed",
    "tqdm",
]
