import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.application.structure.coefficients import (
    load_coefficients_from_file,
    resolve_structure_coefficients,
)


LOGGER = logging.getLogger("test.structure")


def test_resolve_structure_coefficients_prefers_config_values_in_auto_mode(tmp_path):
    elements = pd.Series(["Na", "Cl"])
    cfg_proc = SimpleNamespace(get_coeff=lambda: pd.Series([1.5, 2.5]))
    cells_origin = pd.DataFrame([[0.0, 0.0], [0.5, 0.5]], columns=["x", "y"])

    coeff = resolve_structure_coefficients(
        struct={"dimension": 2},
        cfg_proc=cfg_proc,
        working_path=str(tmp_path),
        elements=elements,
        supercell=np.array([2, 2]),
        vectors=np.eye(2),
        cells_origin=cells_origin,
        logger=LOGGER,
    )

    assert coeff.tolist() == [1.5, 2.5]


def test_load_coefficients_from_file_maps_2d_coeff_matrix(tmp_path):
    coeff_path = tmp_path / "coeff.txt"
    np.savetxt(coeff_path, np.array([[1.0, 2.0], [3.0, 4.0]]))

    coeff = load_coefficients_from_file(
        coeff_path=str(coeff_path),
        dim=2,
        n_atoms=2,
        supercell=np.array([2, 2]),
        vectors=np.eye(2),
        cells_origin=np.array([[0.0, 0.0], [0.5, 0.5]]),
        logger=LOGGER,
    )

    np.testing.assert_allclose(coeff, np.array([1.0, 4.0]))
