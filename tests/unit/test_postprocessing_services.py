import numpy as np

from core.application.postprocessing.decoder_cache import (
    build_decoder_cache_path,
    load_decoder_cache,
    save_decoder_cache,
)
from core.application.postprocessing.loader import resolve_output_dir


class _NoopLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_decoder_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "decoder.npz"
    decoder = np.eye(2)
    save_decoder_cache(str(cache_path), decoder, 2, logger=_NoopLogger())
    loaded_decoder, feature_dim = load_decoder_cache(str(cache_path), logger=_NoopLogger())
    np.testing.assert_allclose(loaded_decoder, decoder)
    assert feature_dim == 2


def test_build_decoder_cache_path_is_stable(tmp_path):
    params = {
        "supercell": np.array([4, 4]),
        "reciprocal_space_intervals_all": [{"h_range": (0.0, 1.0), "k_range": (0.0, 1.0)}],
        "q_window_kind": "cheb",
        "q_window_at_db": 100.0,
        "edge_guard_frac": 0.1,
        "ls_weight_gamma": 0.35,
        "dog_lambda_reg": 1e-3,
    }
    path1 = build_decoder_cache_path(params, str(tmp_path))
    path2 = build_decoder_cache_path(params, str(tmp_path))
    assert path1 == path2


def test_resolve_output_dir_prefers_explicit_path(tmp_path):
    path = resolve_output_dir(None, chunk_id=0, output_dir=str(tmp_path / "out"))
    assert path == str(tmp_path / "out")
