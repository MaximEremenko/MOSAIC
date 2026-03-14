from __future__ import annotations

from dataclasses import dataclass

from core.qspace.intervals.interval_mapping import pad_interval


@dataclass(frozen=True)
class PendingIntervalWork:
    point_rows: list[dict]
    intervals: list[dict]


class IntervalReconstructionService:
    def load_pending_work(self, artifacts, dimension: int) -> PendingIntervalWork:
        unsaved = artifacts.db_manager.get_unsaved_interval_chunks()
        chunk_ids = sorted({chunk_id for _, chunk_id in unsaved})
        interval_ids = sorted({interval_id for interval_id, _ in unsaved})

        point_rows: list[dict] = []
        for chunk_id in chunk_ids:
            point_rows.extend(artifacts.db_manager.get_point_data_for_chunk(chunk_id))

        intervals = [
            pad_interval(interval.to_mapping(dimension), dimension)
            | {"id": interval.interval_id}
            for interval in artifacts.db_manager.get_intervals_by_ids(interval_ids)
        ]
        return PendingIntervalWork(point_rows=point_rows, intervals=intervals)
