from __future__ import annotations

from core.config.values import first_present
from core.processing_mode import normalize_processing_mode
from core.qspace.masking.mask_strategies import DefaultMaskStrategy, EqBasedStrategy
from core.qspace.masking.shape_strategies import (
    CircleShapeStrategy,
    IntervalShapeStrategy,
)


class MaskStrategyService:
    def build(self, dim: int, peak_info: dict, post_mode: str = "displacement"):
        equation = self.get_equation(peak_info)
        if equation is not None:
            return EqBasedStrategy(equation)
        if not self.has_special_points(peak_info):
            return DefaultMaskStrategy()
        if dim == 1:
            return IntervalShapeStrategy(peak_info)
        if dim == 2:
            return CircleShapeStrategy(peak_info)

        r1_val = float(peak_info.get("r1", peak_info.get("radius", 0.1876)))
        r2_val = float(peak_info.get("r2", peak_info.get("radius", 0.2501)))
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
