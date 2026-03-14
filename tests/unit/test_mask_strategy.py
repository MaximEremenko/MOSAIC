import numpy as np

from core.domain.masking.mask_strategies import EqBasedStrategy


def test_eq_based_strategy_masks_points():
    strategy = EqBasedStrategy("h > 0")
    mesh = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mask = strategy.generate_mask(mesh)
    assert mask.tolist() == [True, False]
