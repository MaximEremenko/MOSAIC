#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.runtime.fs_capability import FilesystemCapabilityError, profile_output_filesystem


def _maybe_dask_client(use_dask: bool):
    if not use_dask:
        return None
    try:
        from core.runtime.dask_client import get_client

        return get_client()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate MOSAIC shared-filesystem rename/read visibility."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-digest", default="cross_host_visibility_probe")
    parser.add_argument("--use-dask", action="store_true")
    args = parser.parse_args(argv)

    try:
        manifest = profile_output_filesystem(
            Path(args.output_dir),
            run_digest=args.run_digest,
            client=_maybe_dask_client(bool(args.use_dask)),
            require_cross_host=bool(args.use_dask),
        )
    except FilesystemCapabilityError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "run_digest": manifest.run_digest,
                "capability_digest": manifest.capability_digest,
                "capabilities": manifest.capabilities,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
