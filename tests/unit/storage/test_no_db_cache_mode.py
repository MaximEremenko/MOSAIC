from __future__ import annotations

from core.storage.database_manager import ManifestOnlyDatabaseManager
from core.storage.db_cache import resolve_db_cache_config


def test_no_db_cache_mode_uses_manifest_only_manager(tmp_path):
    config = resolve_db_cache_config(
        output_dir=tmp_path,
        no_db_cache=True,
    )
    db = ManifestOnlyDatabaseManager(dimension=1)

    interval_ids = db.insert_reciprocal_space_interval_batch([{"h_range": (0.0, 1.0)}])
    db.insert_point_data_batch(
        [
            {
                "central_point_id": 10,
                "coordinates": [0.0],
                "dist_from_atom_center": [0.0],
                "step_in_frac": [1.0],
                "chunk_id": 2,
                "grid_amplitude_initialized": 0,
            }
        ]
    )
    db.insert_interval_chunk_status_batch([(interval_ids[0], 2, 0)])

    assert not config.enabled
    assert db.db_path == ":manifest-only:"
    assert db.get_unsaved_interval_chunks() == [(interval_ids[0], 2)]
    assert db.get_point_data_for_chunk(2)[0]["central_point_id"] == 10
    assert not (tmp_path / "point_reciprocal_space_associations.db").exists()


def test_local_db_path_uses_env_or_temp(monkeypatch, tmp_path):
    monkeypatch.setenv("MOSAIC_DB_CACHE_DIR", str(tmp_path / "cache"))

    config = resolve_db_cache_config(
        db_path="local",
        run_digest="run123",
        output_dir=tmp_path,
    )

    assert config.enabled
    assert config.mode == "local"
    assert config.db_path == str(tmp_path / "cache" / "run123.db")
