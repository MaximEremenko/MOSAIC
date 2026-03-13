from __future__ import annotations

from typing import Protocol

from core.domain.models import PointData


class PointProcessor(Protocol):
    def process_parameters(self) -> None:
        ...

    def get_point_data(self) -> PointData:
        ...

    def save_point_data_to_hdf5(self, hdf5_file_path: str) -> None:
        ...

    def load_point_data_from_hdf5(self, hdf5_file_path: str) -> bool:
        ...

