from __future__ import annotations

from typing import Any, Protocol

from core.models import ReciprocalInterval


class PointRepository(Protocol):
    def get_point_data_for_chunk(self, chunk_id: int) -> list[dict[str, Any]]:
        ...

    def get_point_data_for_point_ids(self, point_ids: list[int]) -> list[dict[str, Any]]:
        ...

    def get_pending_chunk_ids(self) -> list[int]:
        ...

    def insert_point_data_batch(self, point_data_list: list[dict[str, Any]]) -> list[int]:
        ...


class IntervalRepository(Protocol):
    def get_pending_parts(self) -> list[dict[str, Any]]:
        ...

    def get_intervals_by_ids(self, interval_ids: list[int]) -> list[ReciprocalInterval]:
        ...

    def insert_reciprocal_space_interval_batch(
        self, interval_list: list[dict[str, Any]]
    ) -> list[int]:
        ...

    def mark_interval_precomputed(self, interval_id: int, done: bool = True) -> None:
        ...

    def is_interval_precomputed(self, interval_id: int) -> bool:
        ...


class ProcessingStateRepository(Protocol):
    def insert_interval_chunk_status_batch(
        self, status_list: list[tuple[int, int, int | bool]]
    ) -> None:
        ...

    def associate_point_reciprocal_space_batch(
        self, associations: list[tuple[int, int]]
    ) -> None:
        ...

    def update_saved_status_for_chunk_or_point(
        self,
        reciprocal_space_id: int,
        point_id: int | None = None,
        chunk_id: int | None = None,
        saved: int = 0,
    ) -> None:
        ...

    def get_unsaved_associations(self) -> list[tuple[int, int]]:
        ...

    def update_interval_chunk_status(
        self, interval_id: int, chunk_id: int, saved: int | bool = 1
    ) -> None:
        ...

    def get_unsaved_interval_chunks(self) -> list[tuple[int, int]]:
        ...


class ParameterSource(Protocol):
    def load(self, run_file: str = "run_parameters.json") -> tuple[Any, Any]:
        ...


class StructureSource(Protocol):
    def load(self, workflow_parameters: Any, working_path: str) -> Any:
        ...


class AmplitudeStore(Protocol):
    def load_data(self, filename: str) -> dict[str, Any]:
        ...

    def save_data(self, data: dict[str, Any], filename: str) -> None:
        ...

    def generate_filename(self, chunk_id: int, suffix: str = "") -> str:
        ...

