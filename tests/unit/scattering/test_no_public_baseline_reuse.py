from __future__ import annotations

import numpy as np
import pytest

from core.scattering.commit import create_scattering_commit_candidate


def test_commit_candidate_ignores_public_or_canonical_payloads_without_attempts(tmp_path):
    np.array([[1.0 + 0.0j, 2.0 + 0.0j]], dtype=np.complex128).tofile(
        tmp_path / "point_data_chunk_3_amplitudes.hdf5"
    )
    public_dir = tmp_path / "processed_point_data"
    public_dir.mkdir()
    (public_dir / "point_data_chunk_3_amplitudes.hdf5").write_bytes(b"not an attempt")

    with pytest.raises(RuntimeError, match="missing attempts"):
        create_scattering_commit_candidate(
            output_dir=tmp_path,
            run_digest="run123",
            chunk_id=3,
            expected_interval_ids=(1,),
        )
