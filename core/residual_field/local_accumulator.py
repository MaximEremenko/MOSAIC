from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.residual_field.contracts import ResidualFieldWorkUnit


@dataclass(frozen=True)
class ResidualFieldLocalAccumulatorPartial:
    work_unit: ResidualFieldWorkUnit
    point_ids: np.ndarray
    grid_shape_nd: np.ndarray
    total_reciprocal_points: int
    contribution_reciprocal_points: int
    amplitudes_delta: np.ndarray
    amplitudes_average: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_ids", np.asarray(self.point_ids, dtype=np.int64).reshape(-1))
        object.__setattr__(self, "grid_shape_nd", np.asarray(self.grid_shape_nd, dtype=np.int64))
        object.__setattr__(self, "amplitudes_delta", np.asarray(self.amplitudes_delta, dtype=np.complex128).reshape(-1))
        object.__setattr__(
            self,
            "amplitudes_average",
            np.asarray(self.amplitudes_average, dtype=np.complex128).reshape(-1),
        )
        object.__setattr__(self, "total_reciprocal_points", int(self.total_reciprocal_points))
        object.__setattr__(
            self,
            "contribution_reciprocal_points",
            int(self.contribution_reciprocal_points),
        )
        if self.amplitudes_delta.shape != self.amplitudes_average.shape:
            raise ValueError("Local residual partial requires matching delta/average shapes.")
        if self.point_ids.shape[0] != self.amplitudes_delta.shape[0]:
            raise ValueError("Local residual partial requires point_ids to match payload length.")

    @property
    def chunk_id(self) -> int:
        return int(self.work_unit.chunk_id)

    @property
    def parameter_digest(self) -> str:
        return str(self.work_unit.parameter_digest)

    @property
    def interval_ids(self) -> tuple[int, ...]:
        if self.work_unit.interval_ids:
            return tuple(int(interval_id) for interval_id in self.work_unit.interval_ids)
        if self.work_unit.interval_id is None:
            return ()
        return (int(self.work_unit.interval_id),)


def estimate_local_accumulator_bytes(partial: ResidualFieldLocalAccumulatorPartial) -> int:
    return int(
        partial.point_ids.nbytes
        + partial.grid_shape_nd.nbytes
        + partial.amplitudes_delta.nbytes
        + partial.amplitudes_average.nbytes
    )


def build_local_accumulator_snapshot_path(
    output_dir: str,
    *,
    chunk_id: int,
    parameter_digest: str,
    partition_id: int | None = None,
    snapshot_seq: int,
) -> Path:
    partition_suffix = (
        f"_partition_{int(partition_id)}" if partition_id is not None else ""
    )
    return (
        Path(output_dir)
        / "residual_shards"
        / f"chunk_{chunk_id}"
        / f"local_accumulator{partition_suffix}_seq_{int(snapshot_seq)}_params_{parameter_digest}.npz"
    )


def load_local_accumulator_snapshot(
    output_dir: str,
    *,
    chunk_id: int,
    parameter_digest: str,
    partition_id: int | None = None,
    snapshot_seq: int,
) -> dict[str, object] | None:
    snapshot_path = build_local_accumulator_snapshot_path(
        output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        partition_id=partition_id,
        snapshot_seq=snapshot_seq,
    )
    if not snapshot_path.exists():
        return None
    with np.load(snapshot_path, allow_pickle=False) as data:
        return {
            "point_ids": np.asarray(data["point_ids"], dtype=np.int64),
            "grid_shape_nd": np.asarray(data["grid_shape_nd"], dtype=np.int64),
            "amplitudes_delta": np.asarray(data["amplitudes_delta"], dtype=np.complex128),
            "amplitudes_average": np.asarray(data["amplitudes_average"], dtype=np.complex128),
            "reciprocal_point_count": int(np.asarray(data["reciprocal_point_count"]).ravel()[0]),
            "total_reciprocal_points": int(np.asarray(data["total_reciprocal_points"]).ravel()[0]),
            "incorporated_interval_ids": tuple(
                int(interval_id) for interval_id in np.asarray(data["incorporated_interval_ids"], dtype=np.int64).tolist()
            ),
            "partition_id": (
                int(np.asarray(data["partition_id"]).ravel()[0])
                if "partition_id" in data
                else (int(partition_id) if partition_id is not None else None)
            ),
            "storage_mode": str(data["storage_mode"].tolist()),
            "checkpoint_write_count": int(
                np.asarray(data["checkpoint_write_count"]).ravel()[0]
            ) if "checkpoint_write_count" in data else int(snapshot_seq),
            "checkpoint_bytes_written_total": int(
                np.asarray(data["checkpoint_bytes_written_total"]).ravel()[0]
            ) if "checkpoint_bytes_written_total" in data else 0,
            "checkpoint_wall_seconds_total": float(
                np.asarray(data["checkpoint_wall_seconds_total"]).ravel()[0]
            ) if "checkpoint_wall_seconds_total" in data else 0.0,
            "checkpoint_cadence_batches": int(
                np.asarray(data["checkpoint_cadence_batches"]).ravel()[0]
            ) if "checkpoint_cadence_batches" in data else 0,
            "point_start": (
                int(np.asarray(data["point_start"]).ravel()[0])
                if "point_start" in data and int(np.asarray(data["point_start"]).ravel()[0]) >= 0
                else None
            ),
            "point_stop": (
                int(np.asarray(data["point_stop"]).ravel()[0])
                if "point_stop" in data and int(np.asarray(data["point_stop"]).ravel()[0]) >= 0
                else None
            ),
        }


def write_local_accumulator_snapshot(
    output_dir: str,
    *,
    chunk_id: int,
    parameter_digest: str,
    partition_id: int | None = None,
    snapshot_seq: int,
    point_ids: np.ndarray,
    grid_shape_nd: np.ndarray,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    reciprocal_point_count: int,
    total_reciprocal_points: int,
    incorporated_interval_ids: tuple[int, ...],
    storage_mode: str,
    checkpoint_write_count: int = 0,
    checkpoint_bytes_written_total: int = 0,
    checkpoint_wall_seconds_total: float = 0.0,
    checkpoint_cadence_batches: int = 0,
    point_start: int | None = None,
    point_stop: int | None = None,
    compress: bool = False,
) -> Path:
    snapshot_path = build_local_accumulator_snapshot_path(
        output_dir,
        chunk_id=chunk_id,
        parameter_digest=parameter_digest,
        partition_id=partition_id,
        snapshot_seq=snapshot_seq,
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=snapshot_path.parent,
        prefix=f"{snapshot_path.stem}_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        save_fn = np.savez_compressed if compress else np.savez
        save_fn(
            handle,
            point_ids=np.asarray(point_ids, dtype=np.int64),
            grid_shape_nd=np.asarray(grid_shape_nd, dtype=np.int64),
            amplitudes_delta=np.asarray(amplitudes_delta, dtype=np.complex128),
            amplitudes_average=np.asarray(amplitudes_average, dtype=np.complex128),
            reciprocal_point_count=np.array([int(reciprocal_point_count)], dtype=np.int64),
            total_reciprocal_points=np.array([int(total_reciprocal_points)], dtype=np.int64),
            incorporated_interval_ids=np.asarray(
                tuple(sorted(set(int(v) for v in incorporated_interval_ids))),
                dtype=np.int64,
            ),
            partition_id=np.array(
                [-1 if partition_id is None else int(partition_id)],
                dtype=np.int64,
            ),
            storage_mode=np.array([str(storage_mode)]),
            checkpoint_write_count=np.array([int(checkpoint_write_count)], dtype=np.int64),
            checkpoint_bytes_written_total=np.array([int(checkpoint_bytes_written_total)], dtype=np.int64),
            checkpoint_wall_seconds_total=np.array([float(checkpoint_wall_seconds_total)], dtype=np.float64),
            checkpoint_cadence_batches=np.array([int(checkpoint_cadence_batches)], dtype=np.int64),
            point_start=np.array([-1 if point_start is None else int(point_start)], dtype=np.int64),
            point_stop=np.array([-1 if point_stop is None else int(point_stop)], dtype=np.int64),
        )
    Path(handle.name).replace(snapshot_path)
    return snapshot_path


def make_local_accumulator_snapshot_key(
    *,
    chunk_id: int,
    parameter_digest: str,
    partition_id: int | None = None,
    snapshot_seq: int,
) -> str:
    partition_token = (
        f":partition-{int(partition_id)}" if partition_id is not None else ""
    )
    return (
        f"local-accumulator-snapshot:chunk-{int(chunk_id)}:"
        f"params-{parameter_digest}{partition_token}:seq-{int(snapshot_seq)}"
    )


def parse_local_accumulator_snapshot_key(
    snapshot_key: str,
) -> tuple[int, str, int | None, int] | None:
    try:
        prefix, chunk_token, params_token, *tail = str(snapshot_key).split(":")
    except ValueError:
        return None
    if prefix != "local-accumulator-snapshot":
        return None
    if not chunk_token.startswith("chunk-"):
        return None
    if not params_token.startswith("params-"):
        return None
    if len(tail) == 1:
        partition_id = None
        seq_token = tail[0]
    elif len(tail) == 2:
        partition_token, seq_token = tail
        if not partition_token.startswith("partition-"):
            return None
        partition_id = int(partition_token.removeprefix("partition-"))
    else:
        return None
    if not seq_token.startswith("seq-"):
        return None
    return (
        int(chunk_token.removeprefix("chunk-")),
        params_token.removeprefix("params-"),
        partition_id,
        int(seq_token.removeprefix("seq-")),
    )


class LiveLocalAccumulator:
    def __init__(
        self,
        *,
        chunk_id: int,
        parameter_digest: str,
        partition_id: int | None,
        point_ids: np.ndarray,
        grid_shape_nd: np.ndarray,
        total_reciprocal_points: int,
        reciprocal_point_count: int,
        amplitudes_delta,
        amplitudes_average,
        incorporated_interval_ids: tuple[int, ...],
        durable_interval_ids: tuple[int, ...],
        durable_snapshot_seq: int,
        storage_mode: str,
        checkpoint_write_count: int = 0,
        checkpoint_bytes_written_total: int = 0,
        checkpoint_wall_seconds_total: float = 0.0,
        checkpoint_cadence_batches: int = 0,
        live_dir: Path | None = None,
        point_start: int | None = None,
        point_stop: int | None = None,
    ) -> None:
        self.chunk_id = int(chunk_id)
        self.parameter_digest = str(parameter_digest)
        self.partition_id = int(partition_id) if partition_id is not None else None
        self.point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
        self.point_start = int(point_start) if point_start is not None else None
        self.point_stop = int(point_stop) if point_stop is not None else None
        self.grid_shape_nd = np.asarray(grid_shape_nd, dtype=np.int64)
        self.total_reciprocal_points = int(total_reciprocal_points)
        self.reciprocal_point_count = int(reciprocal_point_count)
        self.storage_mode = str(storage_mode)
        self.live_dir = live_dir
        self.current_interval_ids = set(int(interval_id) for interval_id in incorporated_interval_ids)
        self.durable_interval_ids = set(int(interval_id) for interval_id in durable_interval_ids)
        self.durable_snapshot_seq = int(durable_snapshot_seq)
        self.accepted_since_snapshot = 0
        self.checkpoint_write_count = int(checkpoint_write_count)
        self.checkpoint_bytes_written_total = int(checkpoint_bytes_written_total)
        self.checkpoint_wall_seconds_total = float(checkpoint_wall_seconds_total)
        self.checkpoint_cadence_batches = int(checkpoint_cadence_batches)
        self.amplitudes_delta = amplitudes_delta
        self.amplitudes_average = amplitudes_average

    @classmethod
    def from_partial(
        cls,
        partial: ResidualFieldLocalAccumulatorPartial,
        *,
        scratch_root: str,
        max_ram_bytes: int,
    ) -> "LiveLocalAccumulator":
        return cls.from_arrays(
            partial.work_unit,
            point_ids=partial.point_ids,
            grid_shape_nd=partial.grid_shape_nd,
            total_reciprocal_points=partial.total_reciprocal_points,
            amplitudes_delta=partial.amplitudes_delta,
            amplitudes_average=partial.amplitudes_average,
            scratch_root=scratch_root,
            max_ram_bytes=max_ram_bytes,
        )

    @classmethod
    def from_arrays(
        cls,
        work_unit: ResidualFieldWorkUnit,
        *,
        point_ids: np.ndarray,
        grid_shape_nd: np.ndarray,
        total_reciprocal_points: int,
        amplitudes_delta: np.ndarray,
        amplitudes_average: np.ndarray,
        scratch_root: str,
        max_ram_bytes: int,
    ) -> "LiveLocalAccumulator":
        point_ids_arr = np.asarray(point_ids, dtype=np.int64).reshape(-1)
        grid_shape_nd_arr = np.asarray(grid_shape_nd, dtype=np.int64)
        amplitudes_delta_arr = np.asarray(amplitudes_delta, dtype=np.complex128).reshape(-1)
        amplitudes_average_arr = np.asarray(amplitudes_average, dtype=np.complex128).reshape(-1)
        if amplitudes_delta_arr.shape != amplitudes_average_arr.shape:
            raise ValueError("Local residual accumulation requires matching delta/average shapes.")
        if point_ids_arr.shape[0] != amplitudes_delta_arr.shape[0]:
            raise ValueError("Local residual accumulation requires point_ids to match payload length.")
        storage_mode = (
            "ram"
            if (
                point_ids_arr.nbytes
                + grid_shape_nd_arr.nbytes
                + amplitudes_delta_arr.nbytes
                + amplitudes_average_arr.nbytes
            )
            <= int(max_ram_bytes)
            else "file"
        )
        amplitudes_delta, amplitudes_average, live_dir = _allocate_live_arrays(
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            partition_id=work_unit.partition_id,
            scratch_root=scratch_root,
            template_delta=amplitudes_delta_arr,
            template_average=amplitudes_average_arr,
            storage_mode=storage_mode,
        )
        amplitudes_delta[:] = 0
        amplitudes_average[:] = 0
        return cls(
            chunk_id=work_unit.chunk_id,
            parameter_digest=work_unit.parameter_digest,
            partition_id=work_unit.partition_id,
            point_ids=point_ids_arr,
            grid_shape_nd=grid_shape_nd_arr,
            total_reciprocal_points=total_reciprocal_points,
            reciprocal_point_count=0,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            incorporated_interval_ids=(),
            durable_interval_ids=(),
            durable_snapshot_seq=0,
            storage_mode=storage_mode,
            checkpoint_write_count=0,
            checkpoint_bytes_written_total=0,
            checkpoint_wall_seconds_total=0.0,
            checkpoint_cadence_batches=0,
            live_dir=live_dir,
            point_start=getattr(work_unit, "point_start", None),
            point_stop=getattr(work_unit, "point_stop", None),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, object],
        *,
        chunk_id: int,
        parameter_digest: str,
        partition_id: int | None,
        snapshot_seq: int,
        scratch_root: str,
        max_ram_bytes: int,
    ) -> "LiveLocalAccumulator":
        storage_mode = (
            str(snapshot["storage_mode"])
            if str(snapshot["storage_mode"]) in {"ram", "file"}
            else "file"
            if int(np.asarray(snapshot["amplitudes_delta"]).nbytes + np.asarray(snapshot["amplitudes_average"]).nbytes) > int(max_ram_bytes)
            else "ram"
        )
        amplitudes_delta, amplitudes_average, live_dir = _allocate_live_arrays(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            partition_id=partition_id,
            scratch_root=scratch_root,
            template_delta=np.asarray(snapshot["amplitudes_delta"], dtype=np.complex128),
            template_average=np.asarray(snapshot["amplitudes_average"], dtype=np.complex128),
            storage_mode=storage_mode,
        )
        incorporated_interval_ids = tuple(
            int(interval_id) for interval_id in snapshot["incorporated_interval_ids"]
        )
        return cls(
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            partition_id=partition_id,
            point_ids=np.asarray(snapshot["point_ids"], dtype=np.int64),
            grid_shape_nd=np.asarray(snapshot["grid_shape_nd"], dtype=np.int64),
            total_reciprocal_points=int(snapshot["total_reciprocal_points"]),
            reciprocal_point_count=int(snapshot["reciprocal_point_count"]),
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            incorporated_interval_ids=incorporated_interval_ids,
            durable_interval_ids=incorporated_interval_ids,
            durable_snapshot_seq=int(snapshot_seq),
            storage_mode=storage_mode,
            checkpoint_write_count=int(snapshot.get("checkpoint_write_count", snapshot_seq)),
            checkpoint_bytes_written_total=int(snapshot.get("checkpoint_bytes_written_total", 0)),
            checkpoint_wall_seconds_total=float(snapshot.get("checkpoint_wall_seconds_total", 0.0)),
            checkpoint_cadence_batches=int(snapshot.get("checkpoint_cadence_batches", 0)),
            live_dir=live_dir,
            point_start=int(snapshot["point_start"]) if snapshot.get("point_start") is not None else None,
            point_stop=int(snapshot["point_stop"]) if snapshot.get("point_stop") is not None else None,
        )

    def should_skip_partial(self, partial: ResidualFieldLocalAccumulatorPartial) -> bool:
        return set(partial.interval_ids).issubset(self.current_interval_ids)

    def should_skip_interval_ids(self, interval_ids: tuple[int, ...]) -> bool:
        return set(int(interval_id) for interval_id in interval_ids).issubset(
            self.current_interval_ids
        )

    def accept_partial(self, partial: ResidualFieldLocalAccumulatorPartial) -> None:
        self.accept_contribution(
            partial.work_unit,
            point_ids=partial.point_ids,
            grid_shape_nd=partial.grid_shape_nd,
            total_reciprocal_points=partial.total_reciprocal_points,
            contribution_reciprocal_points=partial.contribution_reciprocal_points,
            amplitudes_delta=partial.amplitudes_delta,
            amplitudes_average=partial.amplitudes_average,
        )

    def accept_contribution(
        self,
        work_unit: ResidualFieldWorkUnit,
        *,
        point_ids: np.ndarray,
        grid_shape_nd: np.ndarray,
        total_reciprocal_points: int,
        contribution_reciprocal_points: int,
        amplitudes_delta: np.ndarray,
        amplitudes_average: np.ndarray,
    ) -> None:
        self._validate_contribution(
            work_unit,
            point_ids=point_ids,
            grid_shape_nd=grid_shape_nd,
            total_reciprocal_points=total_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
        )
        interval_ids = (
            tuple(int(interval_id) for interval_id in work_unit.interval_ids)
            if work_unit.interval_ids
            else ((int(work_unit.interval_id),) if work_unit.interval_id is not None else ())
        )
        if self.should_skip_interval_ids(interval_ids):
            return
        self.amplitudes_delta += np.asarray(amplitudes_delta, dtype=np.complex128).reshape(-1)
        self.amplitudes_average += np.asarray(amplitudes_average, dtype=np.complex128).reshape(-1)
        self.reciprocal_point_count += int(contribution_reciprocal_points)
        self.current_interval_ids.update(interval_ids)
        self.accepted_since_snapshot += 1

    def _validate_partial(self, partial: ResidualFieldLocalAccumulatorPartial) -> None:
        self._validate_contribution(
            partial.work_unit,
            point_ids=partial.point_ids,
            grid_shape_nd=partial.grid_shape_nd,
            total_reciprocal_points=partial.total_reciprocal_points,
            amplitudes_delta=partial.amplitudes_delta,
            amplitudes_average=partial.amplitudes_average,
        )

    def _validate_contribution(
        self,
        work_unit: ResidualFieldWorkUnit,
        *,
        point_ids: np.ndarray,
        grid_shape_nd: np.ndarray,
        total_reciprocal_points: int,
        amplitudes_delta: np.ndarray,
        amplitudes_average: np.ndarray,
    ) -> None:
        point_ids_arr = np.asarray(point_ids, dtype=np.int64).reshape(-1)
        grid_shape_nd_arr = np.asarray(grid_shape_nd, dtype=np.int64)
        amplitudes_delta_arr = np.asarray(amplitudes_delta, dtype=np.complex128).reshape(-1)
        amplitudes_average_arr = np.asarray(amplitudes_average, dtype=np.complex128).reshape(-1)
        if amplitudes_delta_arr.shape != amplitudes_average_arr.shape:
            raise ValueError("Local residual accumulation requires matching delta/average shapes.")
        if point_ids_arr.shape[0] != amplitudes_delta_arr.shape[0]:
            raise ValueError("Local residual accumulation requires point_ids to match payload length.")
        if int(work_unit.chunk_id) != self.chunk_id:
            raise ValueError("Local accumulator partial chunk_id mismatch.")
        if str(work_unit.parameter_digest) != self.parameter_digest:
            raise ValueError("Local accumulator partial parameter digest mismatch.")
        if work_unit.partition_id != self.partition_id:
            raise ValueError("Local accumulator partial partition_id mismatch.")
        if not np.array_equal(self.point_ids, point_ids_arr):
            raise ValueError("Local accumulator partial point_ids mismatch.")
        if not np.array_equal(self.grid_shape_nd, grid_shape_nd_arr):
            raise ValueError("Local accumulator partial grid_shape_nd mismatch.")
        if int(total_reciprocal_points) != self.total_reciprocal_points:
            raise ValueError("Local accumulator total_reciprocal_points mismatch.")

    def snapshot_payload(self) -> dict[str, object]:
        return {
            "point_ids": self.point_ids.copy(),
            "grid_shape_nd": self.grid_shape_nd.copy(),
            "amplitudes_delta": np.asarray(self.amplitudes_delta, dtype=np.complex128).copy(),
            "amplitudes_average": np.asarray(self.amplitudes_average, dtype=np.complex128).copy(),
            "reciprocal_point_count": int(self.reciprocal_point_count),
            "total_reciprocal_points": int(self.total_reciprocal_points),
            "incorporated_interval_ids": tuple(sorted(self.current_interval_ids)),
            "partition_id": self.partition_id,
            "storage_mode": self.storage_mode,
            "checkpoint_write_count": int(self.checkpoint_write_count),
            "checkpoint_bytes_written_total": int(self.checkpoint_bytes_written_total),
            "checkpoint_wall_seconds_total": float(self.checkpoint_wall_seconds_total),
            "checkpoint_cadence_batches": int(self.checkpoint_cadence_batches),
            "point_start": self.point_start,
            "point_stop": self.point_stop,
        }

    def durable_progress_interval_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.durable_interval_ids))

    def next_snapshot_seq(self) -> int:
        return int(self.durable_snapshot_seq) + 1

    def mark_snapshot_committed(self, snapshot_seq: int) -> tuple[int, ...]:
        newly_durable = tuple(sorted(self.current_interval_ids - self.durable_interval_ids))
        self.durable_interval_ids = set(self.current_interval_ids)
        self.durable_snapshot_seq = int(snapshot_seq)
        self.accepted_since_snapshot = 0
        return newly_durable

    def record_checkpoint_metrics(
        self,
        *,
        bytes_written: int,
        wall_seconds: float,
        checkpoint_cadence_batches: int,
    ) -> None:
        self.checkpoint_write_count += 1
        self.checkpoint_bytes_written_total += int(bytes_written)
        self.checkpoint_wall_seconds_total += float(wall_seconds)
        self.checkpoint_cadence_batches = int(checkpoint_cadence_batches)

    def cleanup_live_files(self) -> None:
        if self.live_dir is None:
            return
        for path in self.live_dir.glob("*.npy"):
            path.unlink(missing_ok=True)
        if self.live_dir.exists() and not any(self.live_dir.iterdir()):
            self.live_dir.rmdir()
        parent = self.live_dir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


def _allocate_live_arrays(
    *,
    chunk_id: int,
    parameter_digest: str,
    partition_id: int | None,
    scratch_root: str,
    template_delta: np.ndarray,
    template_average: np.ndarray,
    storage_mode: str,
) -> tuple[np.ndarray, np.ndarray, Path | None]:
    delta = np.asarray(template_delta, dtype=np.complex128).reshape(-1)
    average = np.asarray(template_average, dtype=np.complex128).reshape(-1)
    if storage_mode == "ram":
        return delta.copy(), average.copy(), None

    live_dir = (
        Path(scratch_root).expanduser()
        / "residual_accumulators"
        / f"chunk_{chunk_id}"
        / (
            f"params_{parameter_digest}"
            if partition_id is None
            else f"params_{parameter_digest}_partition_{int(partition_id)}"
        )
    )
    live_dir.mkdir(parents=True, exist_ok=True)
    delta_path = live_dir / "amplitudes_delta.npy"
    average_path = live_dir / "amplitudes_average.npy"
    delta_mm = np.lib.format.open_memmap(
        delta_path,
        mode="w+",
        dtype=np.complex128,
        shape=delta.shape,
    )
    average_mm = np.lib.format.open_memmap(
        average_path,
        mode="w+",
        dtype=np.complex128,
        shape=average.shape,
    )
    delta_mm[:] = delta
    average_mm[:] = average
    return delta_mm, average_mm, live_dir


__all__ = [
    "LiveLocalAccumulator",
    "ResidualFieldLocalAccumulatorPartial",
    "build_local_accumulator_snapshot_path",
    "estimate_local_accumulator_bytes",
    "load_local_accumulator_snapshot",
    "write_local_accumulator_snapshot",
]
