from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DBCacheConfig:
    enabled: bool
    db_path: str | None
    mode: str


def _mapping_get(mapping: Any, key: str, default=None):
    if mapping is None:
        return default
    getter = getattr(mapping, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(mapping, key, default)


def _runtime_mapping(workflow_parameters) -> Any:
    runtime_info = getattr(workflow_parameters, "runtime_info", {}) or {}
    to_mapping = getattr(runtime_info, "to_mapping", None)
    if callable(to_mapping):
        return to_mapping()
    return runtime_info


def _local_db_path(*, run_digest: str | None) -> str:
    root = Path(os.getenv("MOSAIC_DB_CACHE_DIR") or Path(tempfile.gettempdir()) / "mosaic-db-cache")
    name = f"{run_digest or 'pending'}.db"
    return str(root / name)


def resolve_db_cache_config(
    *,
    run_settings=None,
    workflow_parameters=None,
    cli_args=None,
    run_digest: str | None = None,
    output_dir: str | Path | None = None,
    db_path: str | None = None,
    no_db_cache: bool | None = None,
) -> DBCacheConfig:
    del run_settings
    runtime = _runtime_mapping(workflow_parameters)
    cli_no_db = bool(_mapping_get(cli_args, "no_db_cache", False))
    if no_db_cache is None:
        no_db_cache = cli_no_db or bool(_mapping_get(runtime, "no_db_cache", False))
    if no_db_cache:
        return DBCacheConfig(enabled=False, db_path=None, mode="manifest-only")

    selected = (
        db_path
        or _mapping_get(cli_args, "db_path", None)
        or _mapping_get(runtime, "db_path", None)
        or _mapping_get(runtime, "db_cache_path", None)
    )
    if selected is None:
        if output_dir is None:
            return DBCacheConfig(enabled=True, db_path=None, mode="default")
        return DBCacheConfig(
            enabled=True,
            db_path=str(Path(output_dir) / "point_reciprocal_space_associations.db"),
            mode="default",
        )
    selected = str(selected)
    if selected == ":memory:":
        return DBCacheConfig(enabled=True, db_path=":memory:", mode="memory")
    if selected.strip().lower() == "local":
        return DBCacheConfig(
            enabled=True,
            db_path=_local_db_path(run_digest=run_digest),
            mode="local",
        )
    return DBCacheConfig(enabled=True, db_path=selected, mode="explicit")


__all__ = ["DBCacheConfig", "resolve_db_cache_config"]
