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
        np.savez(
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
        live_dir: Path | None = None,
    ) -> None:
        self.chunk_id = int(chunk_id)
        self.parameter_digest = str(parameter_digest)
        self.partition_id = int(partition_id) if partition_id is not None else None
        self.point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
        self.grid_shape_nd = np.asarray(grid_shape_nd, dtype=np.int64)
        self.total_reciprocal_points = int(total_reciprocal_points)
        self.reciprocal_point_count = int(reciprocal_point_count)
        self.storage_mode = str(storage_mode)
        self.live_dir = live_dir
        self.current_interval_ids = set(int(interval_id) for interval_id in incorporated_interval_ids)
        self.durable_interval_ids = set(int(interval_id) for interval_id in durable_interval_ids)
        self.durable_snapshot_seq = int(durable_snapshot_seq)
        self.accepted_since_snapshot = 0
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
        storage_mode = (
            "ram"
            if estimate_local_accumulator_bytes(partial) <= int(max_ram_bytes)
            else "file"
        )
        amplitudes_delta, amplitudes_average, live_dir = _allocate_live_arrays(
            chunk_id=partial.chunk_id,
            parameter_digest=partial.parameter_digest,
            scratch_root=scratch_root,
            template_delta=partial.amplitudes_delta,
            template_average=partial.amplitudes_average,
            storage_mode=storage_mode,
        )
        amplitudes_delta[:] = 0
        amplitudes_average[:] = 0
        return cls(
            chunk_id=partial.chunk_id,
            parameter_digest=partial.parameter_digest,
            partition_id=partial.work_unit.partition_id,
            point_ids=partial.point_ids,
            grid_shape_nd=partial.grid_shape_nd,
            total_reciprocal_points=partial.total_reciprocal_points,
            reciprocal_point_count=0,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            incorporated_interval_ids=(),
            durable_interval_ids=(),
            durable_snapshot_seq=0,
            storage_mode=storage_mode,
            live_dir=live_dir,
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
            live_dir=live_dir,
        )

    def should_skip_partial(self, partial: ResidualFieldLocalAccumulatorPartial) -> bool:
        return set(partial.interval_ids).issubset(self.current_interval_ids)

    def accept_partial(self, partial: ResidualFieldLocalAccumulatorPartial) -> None:
        self._validate_partial(partial)
        if self.should_skip_partial(partial):
            return
        self.amplitudes_delta += partial.amplitudes_delta
        self.amplitudes_average += partial.amplitudes_average
        self.reciprocal_point_count += int(partial.contribution_reciprocal_points)
        self.current_interval_ids.update(int(interval_id) for interval_id in partial.interval_ids)
        self.accepted_since_snapshot += 1

    def _validate_partial(self, partial: ResidualFieldLocalAccumulatorPartial) -> None:
        if partial.chunk_id != self.chunk_id:
            raise ValueError("Local accumulator partial chunk_id mismatch.")
        if partial.parameter_digest != self.parameter_digest:
            raise ValueError("Local accumulator partial parameter digest mismatch.")
        if partial.work_unit.partition_id != self.partition_id:
            raise ValueError("Local accumulator partial partition_id mismatch.")
        if not np.array_equal(self.point_ids, partial.point_ids):
            raise ValueError("Local accumulator partial point_ids mismatch.")
        if not np.array_equal(self.grid_shape_nd, partial.grid_shape_nd):
            raise ValueError("Local accumulator partial grid_shape_nd mismatch.")
        if partial.total_reciprocal_points != self.total_reciprocal_points:
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
        / f"params_{parameter_digest}"
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
