from __future__ import annotations

from core.config.values import first_present
from core.processing_mode import normalize_processing_mode
from core.qspace.masking.mask_strategies import DefaultMaskStrategy, EqBasedStrategy
from core.qspace.masking.shape_strategies import (
    CircleShapeStrategy,
    IntervalShapeStrategy,
)


class MaskStrategyService:
    def _peak_mapping(self, peak_info) -> dict:
        if hasattr(peak_info, "to_mapping"):
            return peak_info.to_mapping()
        return dict(peak_info or {})

    def build(self, dim: int, peak_info, post_mode: str = "displacement"):
        peak_info_mapping = self._peak_mapping(peak_info)
        equation = self.get_equation(peak_info_mapping)
        if equation is not None:
            return EqBasedStrategy(equation)
        if not self.has_special_points(peak_info_mapping):
            return DefaultMaskStrategy()
        if dim == 1:
            return IntervalShapeStrategy(peak_info_mapping)
        if dim == 2:
            return CircleShapeStrategy(peak_info_mapping)

        r1_val = float(peak_info_mapping.get("r1", peak_info_mapping.get("radius", 0.1876)))
        r2_val = float(peak_info_mapping.get("r2", peak_info_mapping.get("radius", 0.2501)))
        if normalize_processing_mode(post_mode) == "displacement":
            condition = """
                (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) <= ({r1})**2) &
                (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) >= ({r2})**2)
            """.strip().format(r1=r1_val, r2=r2_val)
        else:
            condition = """
                (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2) > ({r1})**2) &
                (((Mod(h,1.0) - 0.5)**2 + (Mod(k,1.0) - 0.5)**2 + (Mod(l,1.0) - 0.5)**2) > ({r2})**2)
            """.strip().format(r1=r1_val, r2=r2_val)
        return EqBasedStrategy(condition)

    def get_equation(self, peak_info: dict) -> str | None:
        return first_present(
            peak_info, ("mask_equation", "maskEquation", "equation", "condition")
        )

    def has_special_points(self, peak_info: dict) -> bool:
        special_points = first_present(peak_info, ("specialPoints", "special_points"))
        return isinstance(special_points, list) and len(special_points) > 0
