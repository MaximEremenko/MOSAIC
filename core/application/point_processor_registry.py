from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from core.domain.interfaces.point_processors import PointProcessor
from core.processors.central_point_processor import CentralPointProcessor
from core.processors.from_average_point_processor import FromAveragePointProcessor
from core.processors.full_list_point_processor import FullListPointProcessor

PointProcessorFactoryFn = Callable[[dict, dict, int], PointProcessor]


@dataclass
class PointProcessorRegistry:
    _registry: dict[str, PointProcessorFactoryFn] = field(default_factory=dict)

    def register(self, method: str, factory_fn: PointProcessorFactoryFn) -> None:
        self._registry[method] = factory_fn

    def create(
        self, method: str, parameters: dict, average_structure: dict | None = None
    ) -> PointProcessor:
        try:
            factory_fn = self._registry[method]
        except KeyError as exc:
            raise ValueError(f"Unsupported point processing method: {method}") from exc
        num_chunks = int(parameters.get("rspace_info", {}).get("num_chunks", 10))
        return factory_fn(parameters, average_structure or {}, num_chunks)


def _default_registry() -> PointProcessorRegistry:
    registry = PointProcessorRegistry()
    registry.register(
        "from_average",
        lambda parameters, average_structure, num_chunks: FromAveragePointProcessor(
            parameters, average_structure, num_chunks=num_chunks
        ),
    )
    registry.register(
        "central",
        lambda parameters, average_structure, num_chunks: CentralPointProcessor(
            parameters, average_structure, num_chunks=num_chunks
        ),
    )
    registry.register(
        "full_list",
        lambda parameters, average_structure, num_chunks: FullListPointProcessor(
            parameters, average_structure, num_chunks=num_chunks
        ),
    )
    return registry


POINT_PROCESSOR_REGISTRY = _default_registry()
