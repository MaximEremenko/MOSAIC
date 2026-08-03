# processors/lammps_processor.py

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.config.parsers.lammps_data_parser import LammpsDataParser
from core.config.readers.lammps_file_reader import LammpsFileReader
from core.structure import CellCalculator

# Standard atomic weights (IUPAC 2021, conventional values), used to map the
# per-type masses of a LAMMPS data file onto element symbols. LAMMPS files
# carry no element names; the Masses section is the only chemical identity
# they hold.
_ATOMIC_WEIGHTS: Dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.95, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42,
    "Ag": 107.87, "Cd": 112.41, "In": 114.82, "Sn": 118.71, "Sb": 121.76,
    "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33,
    "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Sm": 150.36,
    "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93,
    "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49,
    "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22,
    "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.98, "Th": 232.04, "U": 238.03,
}

_MASS_MATCH_TOLERANCE = 0.5
_MASS_AMBIGUITY_MARGIN = 0.2
_BOX_MATCH_TOLERANCE = 1e-8


def _element_from_mass(mass: float) -> str:
    ranked = sorted(
        _ATOMIC_WEIGHTS.items(), key=lambda item: abs(item[1] - mass)
    )
    best_symbol, best_weight = ranked[0]
    best_diff = abs(best_weight - mass)
    if best_diff > _MASS_MATCH_TOLERANCE:
        raise ValueError(
            f"No element matches mass {mass} amu within "
            f"{_MASS_MATCH_TOLERANCE} amu."
        )
    runner_symbol, runner_weight = ranked[1]
    if (
        abs(runner_weight - mass) <= _MASS_MATCH_TOLERANCE
        and abs(runner_weight - mass) - best_diff < _MASS_AMBIGUITY_MARGIN
    ):
        raise ValueError(
            f"Mass {mass} amu is ambiguous between {best_symbol} "
            f"({best_weight}) and {runner_symbol} ({runner_weight})."
        )
    return best_symbol


class LammpsDataProcessor:
    """Configuration processor for LAMMPS data files (amorphous path).

    The box is the unit cell: ``supercell = (1, 1, 1)`` and ``vectors`` are
    the box vectors, so the q-grid degenerates to the box reciprocal
    lattice (pitch 2*pi/L) — exact under PBC. See
    docs/amorphous/masked_correlations.pdf.

    An amorphous configuration has no derivable average structure, so the
    reference is an INPUT: ``average_file_path`` (config key
    ``structure.filename_av`` / ``paths.average_structure_file``) names a
    second LAMMPS file holding the reference configuration, paired to the
    primary file by atom id. Without it, the reference defaults to the
    configuration itself (chemical/participation mode, where
    ``reference_mode: homogeneous`` makes the average channel zero and the
    reference is never consulted).

    Conventions:
    - coordinates are Cartesian Angstrom, WRAPPED into the primary box;
      image flags are irrelevant after wrapping (exp(iq.(r+Ln)) = exp(iq.r)
      on the box reciprocal lattice). Displacements are recovered
      downstream under the minimum image, which the one-cell fractional
      wrap provides.
    - ``cells_origin`` is the per-atom reference site (each atom is its own
      one-atom cell anchored at its reference position). The factorized
      crystal average is meaningless here; amorphous runs must select
      ``reference_mode: direct`` or ``homogeneous``.
    - ``refnumbers`` is 1 for every atom: site classes are per element
      (elementSymbol, referenceNumber=1), which selects all atoms of a
      species as patch centers and gives the decoder one class per species.
    """

    def __init__(self, file_path: str, average_file_path: Optional[str] = None):
        self.file_path = file_path
        self.average_file_path = average_file_path
        self.cell_calculator = CellCalculator()
        self.metadata: dict = {}
        self.supercell: np.ndarray | None = None
        self.vectors: np.ndarray | None = None
        self.metric: Dict | None = None
        self.original_coordinates: pd.DataFrame | None = None
        self.average_coordinates: pd.DataFrame | None = None
        self.elements: pd.Series | None = None
        self.refNumbers: pd.Series | None = None

    def process(self) -> None:
        frame, metadata = self._parse(self.file_path)
        self.metadata = metadata
        lengths = self._box_lengths(metadata)
        self.supercell = np.array([1, 1, 1])
        self.vectors = self.cell_calculator.calculate_vectors(
            [lengths[0], lengths[1], lengths[2], 90.0, 90.0, 90.0]
        )
        self.metric = self.cell_calculator.calculate_metric(self.vectors)

        self.original_coordinates = self._wrapped_coordinates(frame, metadata)
        self.elements = pd.Series(
            [_element_from_mass(metadata["masses"][t]) for t in frame["type"]],
            name="element",
        )
        self.refNumbers = pd.Series(np.ones(len(frame), dtype=int), name="refNumber")

        if self.average_file_path is not None:
            reference_frame, reference_metadata = self._parse(self.average_file_path)
            self._require_matching_reference(
                frame, metadata, reference_frame, reference_metadata
            )
            # Wrap the reference into the PRIMARY box: both coordinate sets
            # must live on the same torus for the one-cell minimum-image
            # displacement downstream.
            self.average_coordinates = self._wrapped_coordinates(
                reference_frame, metadata
            )
        else:
            self.average_coordinates = self.original_coordinates.copy()

    # -- contract getters -------------------------------------------------

    def get_coordinates(self) -> pd.DataFrame:
        return self._require(self.original_coordinates, "original coordinates")

    def get_average_coordinates(self) -> pd.DataFrame:
        return self._require(self.average_coordinates, "average coordinates")

    def get_cells_origin(self) -> pd.DataFrame:
        return self._require(self.average_coordinates, "average coordinates").copy()

    def get_supercell(self) -> np.ndarray:
        return self._require(self.supercell, "supercell")

    def get_elements(self) -> pd.Series:
        return self._require(self.elements, "elements")

    def get_refnumbers(self) -> pd.Series:
        return self._require(self.refNumbers, "reference numbers")

    def get_cell_ids(self):
        return None

    def get_vectors(self) -> Optional[np.ndarray]:
        return self.vectors

    def get_metric(self) -> Dict:
        return self._require(self.metric, "metric")

    # -- internals --------------------------------------------------------

    @staticmethod
    def _parse(path: str) -> tuple[pd.DataFrame, dict]:
        parser = LammpsDataParser()
        frame = parser.parse(LammpsFileReader(path).read())
        metadata = parser.metadata
        tilt = np.asarray(metadata.get("tilt", (0.0, 0.0, 0.0)), dtype=float)
        if np.any(np.abs(tilt) > 0.0):
            raise ValueError(
                f"LAMMPS box in {path} has tilt factors {tuple(tilt)}; only "
                "orthogonal boxes are supported (the reciprocal basis "
                "derivation assumes a diagonal cell)."
            )
        return frame, metadata

    @staticmethod
    def _box_lengths(metadata: dict) -> np.ndarray:
        bounds = metadata["bounds"]
        return np.array([bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z")])

    @staticmethod
    def _wrapped_coordinates(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        bounds = metadata["bounds"]
        wrapped = {}
        for axis in ("x", "y", "z"):
            lo, hi = bounds[axis]
            length = hi - lo
            wrapped[axis] = lo + np.mod(frame[axis].to_numpy(dtype=float) - lo, length)
        return pd.DataFrame(wrapped)

    @staticmethod
    def _require_matching_reference(
        frame: pd.DataFrame,
        metadata: dict,
        reference_frame: pd.DataFrame,
        reference_metadata: dict,
    ) -> None:
        if len(reference_frame) != len(frame):
            raise ValueError(
                f"Reference file holds {len(reference_frame)} atoms, "
                f"primary file {len(frame)}."
            )
        if not np.array_equal(
            frame["id"].to_numpy(), reference_frame["id"].to_numpy()
        ):
            raise ValueError("Reference and primary files disagree on atom ids.")
        if not np.array_equal(
            frame["type"].to_numpy(), reference_frame["type"].to_numpy()
        ):
            raise ValueError("Reference and primary files disagree on atom types.")
        primary = LammpsDataProcessor._box_lengths(metadata)
        reference = LammpsDataProcessor._box_lengths(reference_metadata)
        if np.any(np.abs(primary - reference) > _BOX_MATCH_TOLERANCE):
            raise ValueError(
                f"Reference box {tuple(reference)} does not match the primary "
                f"box {tuple(primary)}; displacements are undefined across "
                "different boxes."
            )

    @staticmethod
    def _require(value, name: str):
        if value is None:
            raise ValueError(
                f"LAMMPS {name} are not available. Ensure that 'process()' "
                "has been called."
            )
        return value


__all__ = ["LammpsDataProcessor"]
