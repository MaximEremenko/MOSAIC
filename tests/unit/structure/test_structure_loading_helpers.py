import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.structure.coefficients import (
    load_coefficients_from_file,
    resolve_structure_coefficients,
)
from core.structure.lattice import apply_lattice_vectors


LOGGER = logging.getLogger("test.structure")


def test_resolve_structure_coefficients_supports_config_scheme(tmp_path):
    elements = pd.Series(["Na", "Cl"])
    cfg_proc = SimpleNamespace(get_coeff=lambda: pd.Series([1.5, 2.5]))
    cells_origin = pd.DataFrame([[0.0, 0.0], [0.5, 0.5]], columns=["x", "y"])

    coeff = resolve_structure_coefficients(
        struct={"dimension": 2, "coeff_scheme": "config"},
        cfg_proc=cfg_proc,
        working_path=str(tmp_path),
        elements=elements,
        supercell=np.array([2, 2]),
        vectors=np.eye(2),
        cells_origin=cells_origin,
        logger=LOGGER,
    )

    assert coeff.tolist() == [1.5, 2.5]


def test_resolve_structure_coefficients_defaults_to_ones_when_unspecified(tmp_path):
    elements = pd.Series(["Na", "Cl"])
    cfg_proc = SimpleNamespace(get_coeff=lambda: None)
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

    np.testing.assert_allclose(coeff, np.array([1.0, 1.0]))


def test_resolve_structure_coefficients_supports_atomic_number_scheme(tmp_path):
    elements = pd.Series(["Na", "Cl"])
    cfg_proc = SimpleNamespace(get_coeff=lambda: None)
    cells_origin = pd.DataFrame([[0.0, 0.0], [0.5, 0.5]], columns=["x", "y"])

    coeff = resolve_structure_coefficients(
        struct={"dimension": 2, "coeff_scheme": "atomic_number"},
        cfg_proc=cfg_proc,
        working_path=str(tmp_path),
        elements=elements,
        supercell=np.array([2, 2]),
        vectors=np.eye(2),
        cells_origin=cells_origin,
        logger=LOGGER,
    )

    np.testing.assert_allclose(coeff, np.array([11.0, 17.0]))


def test_resolve_structure_coefficients_supports_neutron_scheme(tmp_path):
    elements = pd.Series(["Na", "Cl"])
    cfg_proc = SimpleNamespace(get_coeff=lambda: None)
    cells_origin = pd.DataFrame([[0.0, 0.0], [0.5, 0.5]], columns=["x", "y"])

    coeff = resolve_structure_coefficients(
        struct={"dimension": 2, "coeff_scheme": "neutron"},
        cfg_proc=cfg_proc,
        working_path=str(tmp_path),
        elements=elements,
        supercell=np.array([2, 2]),
        vectors=np.eye(2),
        cells_origin=cells_origin,
        logger=LOGGER,
    )

    np.testing.assert_allclose(coeff, np.array([0.363, 0.9577]))


def test_resolve_structure_coefficients_rejects_legacy_neutron_scattering_length_alias(tmp_path):
    elements = pd.Series(["Na", "Cl"])
    cfg_proc = SimpleNamespace(get_coeff=lambda: None)
    cells_origin = pd.DataFrame([[0.0, 0.0], [0.5, 0.5]], columns=["x", "y"])

    with pytest.raises(ValueError, match="Legacy coefficient schemes are no longer supported"):
        resolve_structure_coefficients(
            struct={"dimension": 2, "coeff_scheme": "neutron_scattering_length"},
            cfg_proc=cfg_proc,
            working_path=str(tmp_path),
            elements=elements,
            supercell=np.array([2, 2]),
            vectors=np.eye(2),
            cells_origin=cells_origin,
            logger=LOGGER,
        )


def test_resolve_structure_coefficients_rejects_legacy_auto_scheme(tmp_path):
    elements = pd.Series(["Na", "Cl"])
    cfg_proc = SimpleNamespace(get_coeff=lambda: pd.Series([1.5, 2.5]))
    cells_origin = pd.DataFrame([[0.0, 0.0], [0.5, 0.5]], columns=["x", "y"])

    with pytest.raises(ValueError, match="Legacy coefficient schemes are no longer supported"):
        resolve_structure_coefficients(
            struct={"dimension": 2, "coeff_scheme": "auto"},
            cfg_proc=cfg_proc,
            working_path=str(tmp_path),
            elements=elements,
            supercell=np.array([2, 2]),
            vectors=np.eye(2),
            cells_origin=cells_origin,
            logger=LOGGER,
        )


def test_resolve_structure_coefficients_requires_config_values_for_config_scheme(tmp_path):
    elements = pd.Series(["Na", "Cl"])
    cfg_proc = SimpleNamespace(get_coeff=lambda: None)
    cells_origin = pd.DataFrame([[0.0, 0.0], [0.5, 0.5]], columns=["x", "y"])

    with pytest.raises(ValueError, match="requires coefficient values"):
        resolve_structure_coefficients(
            struct={"dimension": 2, "coeff_scheme": "config"},
            cfg_proc=cfg_proc,
            working_path=str(tmp_path),
            elements=elements,
            supercell=np.array([2, 2]),
            vectors=np.eye(2),
            cells_origin=cells_origin,
            logger=LOGGER,
        )


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


def test_apply_lattice_vectors_handles_small_2d_and_3d_transforms_without_blas():
    coords2 = np.array([[1.0, 2.0], [0.5, 0.25]])
    vectors2 = np.array([[2.0, 0.0], [0.5, 3.0]])
    transformed2 = apply_lattice_vectors(coords2, vectors2)
    np.testing.assert_allclose(
        transformed2,
        np.array(
            [
                [3.0, 6.0],
                [1.125, 0.75],
            ]
        ),
    )

    coords3 = np.array([[1.0, 2.0, 3.0]])
    vectors3 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 2.0, 0.0],
            [0.25, 0.75, 3.0],
        ]
    )
    transformed3 = apply_lattice_vectors(coords3, vectors3)
    np.testing.assert_allclose(transformed3, np.array([[2.75, 6.25, 9.0]]))
