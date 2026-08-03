"""LAMMPS-data ingestion: the amorphous structure adapter.

The box is the unit cell (supercell=(1,1,1)); coordinates are Cartesian
Angstrom wrapped into the box; elements come from the Masses section; the
reference configuration comes from a second data file paired by atom id.
"""

import numpy as np
import pytest

from core.config.file_type import determine_configuration_file_type
from core.config.factories.configuration_factory import (
    ConfigurationProcessorFactoryProvider,
)
from core.config.parsers.lammps_data_parser import LammpsDataParser
from core.config.processors.lammps_processor import (
    LammpsDataProcessor,
    _element_from_mass,
)


def _write_data_file(
    path,
    coords,
    *,
    types=None,
    box=(0.0, 10.0),
    masses=((1, 15.999), (2, 28.085)),
    style_comment=" # charge",
    with_charge=True,
    with_images=False,
    shuffle_seed=None,
    tilt=None,
):
    n = len(coords)
    types = types if types is not None else [1] * n
    order = list(range(n))
    if shuffle_seed is not None:
        order = list(np.random.default_rng(shuffle_seed).permutation(n))
    lines = ["Generated for tests", ""]
    lines.append(f"{n} atoms")
    lines.append(f"{len(masses)} atom types")
    lines.append("")
    for axis in ("x", "y", "z"):
        lines.append(f"{box[0]} {box[1]} {axis}lo {axis}hi")
    if tilt is not None:
        lines.append(f"{tilt[0]} {tilt[1]} {tilt[2]} xy xz yz")
    lines += ["", "Masses", ""]
    for type_id, mass in masses:
        lines.append(f"{type_id} {mass}")
    lines += ["", f"Atoms{style_comment}", ""]
    for i in order:
        x, y, z = coords[i]
        row = [str(i + 1), str(types[i])]
        if with_charge:
            row.append("0")
        row += [repr(x), repr(y), repr(z)]
        if with_images:
            row += ["1", "0", "-2"]
        lines.append(" ".join(row))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_file_type_maps_lammps_extensions():
    assert determine_configuration_file_type("glass.data") == "lammps"
    assert determine_configuration_file_type("glass.lmp") == "lammps"
    assert determine_configuration_file_type("a.rmc6f") == "rmc6f"


def test_factory_registered():
    factory = ConfigurationProcessorFactoryProvider.get_factory("lammps")
    proc = factory.create_processor("some.data", "calculate", None)
    assert isinstance(proc, LammpsDataProcessor)


def test_parser_sorts_by_atom_id_and_reads_masses(tmp_path):
    coords = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    path = _write_data_file(
        tmp_path / "a.data", coords, types=[1, 2, 1], shuffle_seed=3
    )
    parser = LammpsDataParser()
    frame = parser.parse(path.read_text())
    assert list(frame["id"]) == [1, 2, 3]
    assert list(frame["type"]) == [1, 2, 1]
    np.testing.assert_allclose(frame[["x", "y", "z"]].to_numpy(), coords)
    assert parser.metadata["masses"] == {1: 15.999, 2: 28.085}
    assert parser.metadata["atom_style"] == "charge"


def test_parser_accepts_image_flags_and_atomic_style(tmp_path):
    coords = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    path = _write_data_file(
        tmp_path / "a.data",
        coords,
        style_comment="",
        with_charge=False,
        with_images=True,
    )
    frame = LammpsDataParser().parse(path.read_text())
    np.testing.assert_allclose(frame[["x", "y", "z"]].to_numpy(), coords)


def test_parser_rejects_atom_count_mismatch(tmp_path):
    path = _write_data_file(tmp_path / "a.data", [(1.0, 2.0, 3.0)] * 2)
    content = path.read_text().replace("2 atoms", "3 atoms")
    with pytest.raises(ValueError, match="declares 3 atoms"):
        LammpsDataParser().parse(content)


def test_element_inference_from_mass():
    assert _element_from_mass(15.999) == "O"
    assert _element_from_mass(28.085) == "Si"
    with pytest.raises(ValueError, match="No element matches"):
        _element_from_mass(200000.0)


def test_processor_wraps_into_box_and_synthesizes_glass_members(tmp_path):
    # one atom outside the box on each side
    coords = [(-0.5, 2.0, 3.0), (10.5, 5.0, 6.0), (7.0, 8.0, 9.0)]
    path = _write_data_file(tmp_path / "a.data", coords, types=[1, 1, 2])
    proc = LammpsDataProcessor(str(path))
    proc.process()
    xyz = proc.get_coordinates().to_numpy()
    assert xyz.min() >= 0.0 and xyz.max() < 10.0
    np.testing.assert_allclose(xyz[0], [9.5, 2.0, 3.0])
    np.testing.assert_allclose(xyz[1], [0.5, 5.0, 6.0])
    np.testing.assert_array_equal(proc.get_supercell(), [1, 1, 1])
    np.testing.assert_allclose(np.diag(proc.get_vectors()), [10.0, 10.0, 10.0])
    assert list(proc.get_elements()) == ["O", "O", "Si"]
    assert set(proc.get_refnumbers()) == {1}
    assert proc.get_cell_ids() is None
    # no reference file: average is the configuration itself,
    # cells_origin is the per-atom reference site
    np.testing.assert_allclose(proc.get_average_coordinates().to_numpy(), xyz)
    np.testing.assert_allclose(proc.get_cells_origin().to_numpy(), xyz)


def test_processor_pairs_reference_by_atom_id(tmp_path):
    reference = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    displaced = [(1.05, 2.0, 3.0), (4.0, 5.1, 6.0)]
    ref_path = _write_data_file(tmp_path / "ref.data", reference, types=[1, 2])
    # displaced file rows deliberately in scrambled order
    disp_path = _write_data_file(
        tmp_path / "disp.data", displaced, types=[1, 2], shuffle_seed=11
    )
    proc = LammpsDataProcessor(str(disp_path), average_file_path=str(ref_path))
    proc.process()
    np.testing.assert_allclose(
        proc.get_coordinates().to_numpy(), displaced, atol=1e-12
    )
    np.testing.assert_allclose(
        proc.get_average_coordinates().to_numpy(), reference, atol=1e-12
    )
    delta = proc.get_coordinates().to_numpy() - proc.get_average_coordinates().to_numpy()
    np.testing.assert_allclose(delta[0], [0.05, 0.0, 0.0], atol=1e-12)


def test_processor_rejects_mismatched_reference(tmp_path):
    main_path = _write_data_file(
        tmp_path / "a.data", [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], types=[1, 2]
    )
    bad_types = _write_data_file(
        tmp_path / "b.data", [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], types=[2, 1]
    )
    proc = LammpsDataProcessor(str(main_path), average_file_path=str(bad_types))
    with pytest.raises(ValueError, match="atom types"):
        proc.process()
    bad_box = _write_data_file(
        tmp_path / "c.data",
        [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        types=[1, 2],
        box=(0.0, 11.0),
    )
    proc = LammpsDataProcessor(str(main_path), average_file_path=str(bad_box))
    with pytest.raises(ValueError, match="does not match the primary box"):
        proc.process()


def test_processor_rejects_tilted_box(tmp_path):
    path = _write_data_file(
        tmp_path / "a.data", [(1.0, 2.0, 3.0)], types=[1], tilt=(0.5, 0.0, 0.0)
    )
    with pytest.raises(ValueError, match="tilt factors"):
        LammpsDataProcessor(str(path)).process()
