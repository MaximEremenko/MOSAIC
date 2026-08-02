"""The identity of the atomic structure a run is computing from.

Every amplitude in the pipeline is a sum over these coordinates, so this
digest belongs in any identity that addresses reusable work. It was always
meant to: ``structure_content_digest`` is the FIRST entry of
``scattering.planning._SCIENTIFIC_KEYS`` and the first key
``build_source_structure_digest`` looks for — nothing ever populated it,
so `run_digest` (and everything addressed under it) was invariant to the
coordinates. Measured on CaTiO3-small: replacing the structure file in
place and re-running the same output directory republished the previous
structure's displacements BYTE-IDENTICALLY, because the run tree, the
chunk commits and the residual progress manifests all resolved to the same
addresses.

Computed once, at structure load, and handed to every stage — one run, one
structure identity. It hashes the parsed arrays rather than the file bytes
so that a re-exported file with identical coordinates keeps its
checkpoints, and it must be STABLE across processes: the run tree is
addressed by it, so a digest that drifted between invocations would send
every restart to a fresh family and destroy resume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.storage.atomic import atomic_write_json
from core.storage.digests import digest_dict, normalize_digest_input

STRUCTURE_IDENTITY_SCHEMA_VERSION = 1

# The arrays a scattering amplitude is a function of. `average_coords` and
# `coeff` are included because the residual/displacement path differences
# them against the instantaneous coordinates.
_STRUCTURE_KEYS = (
    "original_coords",
    "average_coords",
    "cells_origin",
    "elements",
    "refnumbers",
    "vectors",
    "supercell",
    "coeff",
)


def _digestable(value: Any) -> Any:
    """Reduce a structure member to something the digest normalizer takes.

    ``coeff`` arrives as a pandas DataFrame (and other members can arrive
    as Series), which the normalizer passes straight through to
    ``json.dumps`` — a TypeError at the very first stage of every run.
    Column labels are digested alongside the values so two frames with the
    same numbers under different columns cannot collide."""
    if value is None:
        return None
    columns = getattr(value, "columns", None)
    to_numpy = getattr(value, "to_numpy", None)
    if callable(to_numpy) and not isinstance(value, np.ndarray):
        values = np.asarray(to_numpy())
        if columns is not None:
            return {
                "columns": [str(name) for name in columns],
                "values": values,
            }
        return values
    return value


def build_structure_content_digest(source: Mapping[str, Any]) -> str:
    """Digest the structure content from a mapping of its arrays.

    Missing keys are digested as ``None`` rather than skipped, so a run
    that omits an array can never collide with one that supplies it."""
    payload: dict[str, Any] = {
        "schema_version": STRUCTURE_IDENTITY_SCHEMA_VERSION,
    }
    for key in _STRUCTURE_KEYS:
        payload[key] = _digestable(source.get(key))
    return digest_dict(
        normalize_digest_input(payload),
        domain="mosaic.structure.content.v1",
    )


def structure_content_digest_from_structure(structure: Any) -> str:
    """Digest a loaded :class:`StructureData` (the workflow's copy).

    A member this function cannot find is an ERROR, not a None: reading it
    with a default is how ``structure_content_digest`` came to be declared
    everywhere and populated nowhere, and a renamed field would silently
    drop out of the identity that addresses the whole run tree."""
    missing = [key for key in _STRUCTURE_KEYS if not hasattr(structure, key)]
    if missing:
        raise AttributeError(
            f"{type(structure).__name__} is missing structure identity "
            f"member(s) {', '.join(missing)}. The digest addresses durable "
            "state, so a member it cannot see must not be digested as absent."
        )
    return build_structure_content_digest(
        {key: getattr(structure, key) for key in _STRUCTURE_KEYS}
    )


STRUCTURE_IDENTITY_FILENAME = "structure_identity.json"


class StructureIdentityConflict(RuntimeError):
    """The output directory was built from a different structure."""


def structure_identity_path(output_dir: Any) -> Path:
    return Path(output_dir) / ".mosaic" / STRUCTURE_IDENTITY_FILENAME


def enforce_output_dir_structure(output_dir: Any, structure_digest: str) -> None:
    """Refuse to reuse an output directory built from another structure.

    Reuse in this pipeline is decided in many places — the run tree, the
    reducer-progress manifests, the decoder commit, the point data, the
    loose residual artifacts — and most of them historically asked only
    whether a file EXISTS. Making each one structure-aware individually
    leaves the next one that is added exposed by default. This is the one
    check that cannot be bypassed by adding a new cache: the directory as
    a whole belongs to exactly one structure.

    First run in a directory records the identity; a matching run proceeds;
    a mismatching run stops before computing anything. ``fresh_start``
    removes the output directory (and this record with it), which is the
    supported way to point one directory at a new structure."""
    path = structure_identity_path(output_dir)
    digest = str(structure_digest)
    if path.exists():
        try:
            recorded = json.loads(path.read_text(encoding="utf-8"))
            previous = str(recorded.get("source_structure_digest", ""))
        except Exception as exc:
            # Fail CLOSED. This record is the only thing standing between a
            # reused directory and another structure's results, and it is
            # written atomically (temp -> fsync -> rename), so an unreadable
            # one is not an ordinary torn write. Treating it as "no record"
            # would let the very case it guards against proceed silently.
            raise StructureIdentityConflict(
                f"Output directory {Path(output_dir)} has an unreadable "
                f"structure identity record {path} ({exc}). It cannot be "
                "shown to belong to this structure. Set "
                "processing.fresh_start=true to rebuild the directory, or "
                "remove that file if you are certain the contents match."
            ) from exc
        if not previous:
            raise StructureIdentityConflict(
                f"Output directory {Path(output_dir)} has a structure "
                f"identity record {path} with no digest in it; it cannot be "
                "shown to belong to this structure. Set "
                "processing.fresh_start=true to rebuild the directory."
            )
        if previous != digest:
            raise StructureIdentityConflict(
                f"Output directory {Path(output_dir)} holds results for a "
                f"different structure (recorded {previous[:16]}…, current "
                f"{digest[:16]}…). Re-running here would mix them: the "
                "residual artifacts, decoder cache and point data on disk "
                "belong to the recorded structure, and the layers that "
                "decide reuse cannot all tell them apart. Set "
                "processing.fresh_start=true to rebuild this directory, or "
                "point paths.output_directory at a new one."
            )
        if previous == digest:
            return
    atomic_write_json(
        path,
        {
            "schema": "mosaic.structure_identity",
            "schema_version": STRUCTURE_IDENTITY_SCHEMA_VERSION,
            "source_structure_digest": digest,
        },
        indent=2,
    )


__all__ = [
    "STRUCTURE_IDENTITY_FILENAME",
    "STRUCTURE_IDENTITY_SCHEMA_VERSION",
    "StructureIdentityConflict",
    "build_structure_content_digest",
    "enforce_output_dir_structure",
    "structure_content_digest_from_structure",
    "structure_identity_path",
]
