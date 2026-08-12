"""Shared environment-variable parsing helpers.

Consolidates the per-module ``_env_int``/``_env_bool`` copies (dask_client,
mask_strategies, nufft_policy, scattering grid, cunufft wrapper). Invalid
values fall back to ``default``; pass ``logger`` (and optionally ``level``)
to report ignored values. ``source`` lets callers inject a mapping instead
of ``os.environ`` (nufft_policy's testable resolution path).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

__all__ = ["env_bool", "env_int"]

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def env_int(
    name: str,
    default: int = 0,
    *,
    source: Mapping[str, str] | None = None,
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
) -> int:
    env = os.environ if source is None else source
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        if logger is not None:
            logger.log(level, "Ignoring invalid integer %s=%r", name, raw)
        return int(default)


def env_bool(
    name: str,
    default: bool | None = False,
    *,
    source: Mapping[str, str] | None = None,
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
) -> bool | None:
    env = os.environ if source is None else source
    raw = env.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    if logger is not None:
        logger.log(level, "Ignoring invalid boolean %s=%r; using %s", name, raw, default)
    return default
