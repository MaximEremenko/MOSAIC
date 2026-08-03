# parsers/lammps_data_parser.py

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Section keywords that may follow the header of a LAMMPS data file. Only
# ``Masses`` and ``Atoms`` are consumed; every other section is skipped.
_SECTION_KEYWORDS = (
    "Masses",
    "Atoms",
    "Velocities",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Pair Coeffs",
    "PairIJ Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
)

# Column layouts by atom_style. Image flags (ix iy iz) may follow each row.
_ATOM_STYLE_COLUMNS = {
    "charge": ("id", "type", "q", "x", "y", "z"),
    "atomic": ("id", "type", "x", "y", "z"),
}


class LammpsDataParser:
    """Parser for LAMMPS ``write_data`` files (atom_style atomic/charge).

    Returns the header metadata (atom counts, box bounds, tilt factors,
    per-type masses, atom style) and the Atoms table SORTED BY ATOM ID.
    ``write_data`` emits atoms in spatial-decomposition order, which varies
    with the MPI grid; id order is the deterministic, physical ordering and
    is what lets a displaced file and its reference pair up row by row.
    """

    def __init__(self) -> None:
        self.metadata: dict = {}

    def parse(self, content: str) -> pd.DataFrame:
        lines = content.splitlines()
        self.metadata = {}
        tilt = (0.0, 0.0, 0.0)
        bounds: dict[str, tuple[float, float]] = {}
        masses: dict[int, float] = {}
        atom_style_hint: str | None = None
        atom_rows: list[list[float]] = []

        section = None
        section_pending_blank = False
        for index, raw in enumerate(lines):
            if index == 0:
                continue  # title line
            line = raw.split("#", 1)[0].strip()
            comment = raw.split("#", 1)[1].strip() if "#" in raw else ""

            keyword = _section_keyword(raw)
            if keyword is not None:
                section = keyword
                section_pending_blank = True
                if keyword == "Atoms" and comment:
                    atom_style_hint = comment.split()[0].lower()
                continue
            if not line:
                if section is not None and not section_pending_blank:
                    section = None
                continue
            section_pending_blank = False

            if section is None:
                self._parse_header_line(line, bounds)
                if line.endswith("xy xz yz"):
                    parts = line.split()
                    tilt = (float(parts[0]), float(parts[1]), float(parts[2]))
                continue
            if section == "Masses":
                parts = line.split()
                masses[int(parts[0])] = float(parts[1])
                continue
            if section == "Atoms":
                atom_rows.append([float(token) for token in line.split()])
                continue
            # any other section: skip rows

        n_atoms = self.metadata.get("n_atoms")
        if n_atoms is None:
            raise ValueError("LAMMPS data file has no '<N> atoms' header line.")
        if not atom_rows:
            raise ValueError("LAMMPS data file has no Atoms section.")
        if len(atom_rows) != int(n_atoms):
            raise ValueError(
                f"Atoms section holds {len(atom_rows)} rows but the header "
                f"declares {int(n_atoms)} atoms."
            )
        for axis in ("x", "y", "z"):
            if axis not in bounds:
                raise ValueError(f"Missing '{axis}lo {axis}hi' box bounds.")
        if not masses:
            raise ValueError(
                "LAMMPS data file has no Masses section; element symbols "
                "are inferred from masses and cannot be resolved without it."
            )

        atom_style, columns = _resolve_atom_style(
            atom_style_hint, len(atom_rows[0])
        )
        frame = pd.DataFrame(atom_rows, columns=list(columns) + _image_columns(len(atom_rows[0]), columns))
        frame["id"] = frame["id"].astype(int)
        frame["type"] = frame["type"].astype(int)
        frame = frame.sort_values("id").reset_index(drop=True)

        self.metadata.update(
            {
                "bounds": bounds,
                "tilt": tilt,
                "masses": masses,
                "atom_style": atom_style,
            }
        )
        return frame

    def _parse_header_line(self, line: str, bounds: dict) -> None:
        match = re.match(r"^(-?[\d.eE+-]+)\s+atoms$", line)
        if match:
            self.metadata["n_atoms"] = int(float(match.group(1)))
            return
        match = re.match(r"^(-?[\d.eE+-]+)\s+atom types$", line)
        if match:
            self.metadata["n_types"] = int(float(match.group(1)))
            return
        for axis in ("x", "y", "z"):
            if line.endswith(f"{axis}lo {axis}hi"):
                parts = line.split()
                bounds[axis] = (float(parts[0]), float(parts[1]))
                return


def _section_keyword(raw_line: str) -> str | None:
    stripped = raw_line.split("#", 1)[0].strip()
    return stripped if stripped in _SECTION_KEYWORDS else None


def _resolve_atom_style(hint: str | None, n_columns: int) -> tuple[str, tuple]:
    if hint is not None:
        columns = _ATOM_STYLE_COLUMNS.get(hint)
        if columns is None:
            raise ValueError(
                f"Unsupported LAMMPS atom_style '{hint}'; supported: "
                f"{sorted(_ATOM_STYLE_COLUMNS)}."
            )
        if n_columns not in (len(columns), len(columns) + 3):
            raise ValueError(
                f"Atoms row has {n_columns} columns; atom_style '{hint}' "
                f"expects {len(columns)} (+3 image flags)."
            )
        return hint, columns
    # No style comment: infer from the column count. charge (6/9) and
    # atomic (5/8) never overlap, so the count is unambiguous.
    for style, columns in _ATOM_STYLE_COLUMNS.items():
        if n_columns in (len(columns), len(columns) + 3):
            return style, columns
    raise ValueError(
        f"Cannot infer atom_style from a {n_columns}-column Atoms row; "
        "supported styles: atomic (5/8 columns), charge (6/9 columns)."
    )


def _image_columns(n_columns: int, base_columns: tuple) -> list[str]:
    return ["ix", "iy", "iz"] if n_columns == len(base_columns) + 3 else []


__all__ = ["LammpsDataParser"]
