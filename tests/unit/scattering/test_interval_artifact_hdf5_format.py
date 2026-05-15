from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from core.scattering.artifacts import persist_precomputed_interval_artifact
from core.scattering.contracts import ScatteringWorkUnit
from core.scattering.kernels import IntervalTask
from core.scattering.tasks import load_interval_task_payload
from core.storage.database_manager import DatabaseManager


def _sample_task(seed: int = 0) -> IntervalTask:
    rng = np.random.default_rng(seed)
    q_grid = rng.standard_normal((17, 3)).astype(np.float64)
    q_amp = rng.standard_normal((17,)).astype(np.complex128)
    q_amp_av = rng.standard_normal((17,)).astype(np.complex128)
    return IntervalTask(1, "Si", q_grid, q_amp, q_amp_av)


def test_interval_artifact_hdf5_roundtrip_and_digest(tmp_path):
    db = DatabaseManager(str(tmp_path / "state.db"), dimension=3)
    try:
        task = _sample_task()
        work_unit = ScatteringWorkUnit.precompute_interval(
            interval_id=task.irecip_id,
            dimension=3,
            output_dir=str(tmp_path),
        )

        manifest = persist_precomputed_interval_artifact(
            work_unit,
            task,
            db_path=db.db_path,
        )
        path = Path(manifest.artifacts[0].path)

        loaded = load_interval_task_payload(path)
        assert path.suffix == ".hdf5"
        assert loaded.irecip_id == task.irecip_id
        assert loaded.element == task.element
        np.testing.assert_array_equal(loaded.q_grid, task.q_grid)
        np.testing.assert_array_equal(loaded.q_amp, task.q_amp)
        np.testing.assert_array_equal(loaded.q_amp_av, task.q_amp_av)
        with h5py.File(path, "r") as h5file:
            assert "q_grid_digest" in h5file
            expected_digest = h5file["q_grid_digest"][()]
            if isinstance(expected_digest, bytes):
                expected_digest = expected_digest.decode("ascii")
            assert loaded.q_grid_digest == expected_digest
            assert h5file.attrs["schema_version"] == 2
    finally:
        db.close()


def test_interval_loader_rejects_npz_intermediate_payloads(tmp_path):
    task = _sample_task()
    path = tmp_path / "interval_1.npz"
    np.savez(
        path,
        irecip_id=task.irecip_id,
        element=task.element,
        q_grid=task.q_grid,
        q_amp=task.q_amp,
        q_amp_av=task.q_amp_av,
    )

    with pytest.raises(ValueError, match="must be HDF5"):
        load_interval_task_payload(path)
