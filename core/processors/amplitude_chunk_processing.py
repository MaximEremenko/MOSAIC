from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from dask.distributed import Client
from tqdm.contrib.logging import logging_redirect_tqdm

from core.processors.amplitude_grid import (
    IntervalTask,
    _point_list_to_recarray,
    _process_chunk,
)
from core.processors.amplitude_persistence import save_amplitudes_and_meta
from core.processors.amplitude_runtime import (
    DEFAULT_TASK_RETRIES,
    _tqdm,
    _yield_futures_with_results,
)
from core.utilities.cunufft_wrapper import execute_inverse_cunufft, set_cpu_only


logger = logging.getLogger(__name__)


def _task_key(iv_id: int, chunk_id: int) -> str:
    return f"proc-{iv_id}-{chunk_id}"


def _parse_task_key(key: str) -> tuple[int, int]:
    _, iv_id, chunk_id = key.split("-", 2)
    return int(iv_id), int(chunk_id)


def _process_chunk_id(
    chunk_id: int,
    iv_path: Path,
    atoms: np.recarray,
    total_reciprocal_points: int,
    point_data_processor,
    db_path: str,
    quiet_logs: bool = False,
) -> bool:
    if quiet_logs:
        for name in (
            __name__,
            "core.managers.database_manager",
            "DatabaseManager",
            "core.processors.point_data_processor",
            "PointDataProcessor",
            "RIFFTInDataSaver",
        ):
            try:
                logging.getLogger(name).setLevel(logging.WARNING)
            except Exception:
                pass

    recip_id = None
    try:
        with np.load(iv_path, mmap_mode="r") as dat:
            task = IntervalTask(
                int(dat["irecip_id"]),
                str(dat["element"]),
                dat["q_grid"],
                dat["q_amp"],
                dat["q_amp_av"],
            )
        recip_id = task.irecip_id

        chunk_data = [
            {
                "coordinates": atoms["coordinates"][i],
                "dist_from_atom_center": atoms["dist_from_atom_center"][i],
                "step_in_frac": atoms["step_in_frac"][i],
            }
            for i in range(atoms.shape[0])
        ]
        rifft_grid, grid_shape = _process_chunk(chunk_data)
        amplitudes_delta = execute_inverse_cunufft(
            q_coords=task.q_grid,
            c=task.q_amp - task.q_amp_av,
            real_coords=rifft_grid,
            eps=1e-12,
        )
        amplitudes_average = execute_inverse_cunufft(
            q_coords=task.q_grid,
            c=task.q_amp_av,
            real_coords=rifft_grid,
            eps=1e-12,
        )
        save_amplitudes_and_meta(
            chunk_id=chunk_id,
            task=task,
            grid_shape_nd=grid_shape,
            total_reciprocal_points=total_reciprocal_points,
            amplitudes_delta=amplitudes_delta,
            amplitudes_average=amplitudes_average,
            point_data_processor=point_data_processor,
            db_path=db_path,
            quiet_logs=quiet_logs,
        )
        return True
    except Exception as err:
        logger.error(
            "chunk %d | iv %s FAILED: %s",
            chunk_id,
            recip_id if recip_id is not None else "n/a",
            err,
            exc_info=True,
        )
        try:
            msg = str(err).lower()
            is_gpu_err = any(
                keyword in msg
                for keyword in (
                    "cuda",
                    "cudart",
                    "cufft",
                    "cufinufft",
                    "cupy",
                    "device-side assert",
                    "illegal memory access",
                    "out of memory",
                    "driver shutting down",
                )
            )
            if is_gpu_err:
                from distributed import get_worker

                try:
                    set_cpu_only(True)
                    worker = get_worker()
                    logger.warning("Worker %s set to CPU-only after GPU error", worker.address)
                except Exception:
                    pass
                try:
                    count = getattr(worker, "gpu_fail_count", 0)
                    setattr(worker, "gpu_fail_count", int(count) + 1)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            import cupy as _cp  # type: ignore

            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return False


def process_chunks_with_intervals(
    interval_files: Iterable[Path],
    *,
    chunk_ids: Iterable[int],
    total_reciprocal_points: int,
    point_data_list: list[dict],
    point_data_processor,
    db_manager,
    client: Client | None,
    max_inflight: int = 5_000,
) -> None:
    unsaved = set(db_manager.get_unsaved_interval_chunks())
    total_tasks = len(unsaved)
    if total_tasks == 0:
        logger.info("Stage-2 skipped – no unsaved (interval, chunk) pairs.")
        return

    if client is None:
        rec = _point_list_to_recarray(point_data_list)
        with _tqdm(total_tasks, desc="Stage 2 (chunks × intervals)", unit="pairs") as pbar:
            for path in interval_files:
                interval_id = int(path.stem.split("_")[1])
                for chunk_id in chunk_ids:
                    if (interval_id, chunk_id) not in unsaved:
                        continue
                    atoms = rec[rec.chunk_id == chunk_id]
                    ok = _process_chunk_id(
                        chunk_id,
                        path,
                        atoms,
                        total_reciprocal_points,
                        point_data_processor,
                        db_manager.db_path,
                        False,
                    )
                    pbar.update(1)
                    pbar.refresh()
                    if not ok:
                        logger.error(
                            "GAVE UP after retries | iv %d | chunk %d (sync)",
                            interval_id,
                            chunk_id,
                        )
        logger.info("Stage-2 finished (sync).")
        return

    fail_streak, fail_threshold = 0, 3
    gpu_tripped = False

    def _trip_to_cpu_only():
        nonlocal gpu_tripped, max_inflight
        if gpu_tripped:
            return
        if hasattr(client, "run"):
            try:
                client.run(set_cpu_only, True)
            except Exception:
                pass
        max_inflight = min(max_inflight, 256)
        gpu_tripped = True
        logger.warning("Circuit-breaker: switching Stage-2 to CPU-only & throttling.")

    rec = _point_list_to_recarray(point_data_list)
    chunk_futs = {
        chunk_id: client.scatter(rec[rec.chunk_id == chunk_id], broadcast=False, hash=False)
        for chunk_id in chunk_ids
    }
    pd_future = client.scatter(point_data_processor, broadcast=True)
    db_path = db_manager.db_path

    retries_left = {key: DEFAULT_TASK_RETRIES for key in unsaved}
    flying: set = set()
    fut_meta: dict = {}
    submitted = 0

    def _submit(iv_path_future, interval_id, chunk_id):
        nonlocal submitted
        fut = client.submit(
            _process_chunk_id,
            chunk_id,
            iv_path_future,
            chunk_futs[chunk_id],
            total_reciprocal_points,
            pd_future,
            db_path,
            True,
            key=_task_key(interval_id, chunk_id),
            pure=False,
            resources={"nufft": 1},
            retries=DEFAULT_TASK_RETRIES,
        )
        flying.add(fut)
        fut_meta[fut] = (interval_id, chunk_id, iv_path_future)
        submitted += 1

    def _harvest_finished_nonblocking(bump):
        nonlocal fail_streak
        done_now = [future for future in list(flying) if future.done()]
        for future in done_now:
            try:
                ok = bool(future.result())
            except Exception:
                ok = False

            flying.discard(future)
            interval_id, chunk_id, ivpf = fut_meta.pop(future, (None, None, None))
            bump()

            if not ok and interval_id is not None:
                fail_streak += 1
                if fail_streak >= fail_threshold:
                    _trip_to_cpu_only()
                if retries_left.get((interval_id, chunk_id), 0) > 0:
                    retries_left[(interval_id, chunk_id)] -= 1
                    _submit(ivpf, interval_id, chunk_id)
            else:
                fail_streak = 0

    with logging_redirect_tqdm():
        with _tqdm(total_tasks, desc="Stage 2 (chunks × intervals)", unit="pairs") as pbar:

            def bump():
                pbar.update(1)
                pbar.refresh()

            for path in interval_files:
                interval_id = int(path.stem.split("_")[1])
                iv_path_future = client.scatter(path, broadcast=False)
                for chunk_id in chunk_ids:
                    if (interval_id, chunk_id) not in unsaved:
                        continue
                    _submit(iv_path_future, interval_id, chunk_id)
                    _harvest_finished_nonblocking(bump)
                    while len(flying) >= max_inflight:
                        for future, ok in _yield_futures_with_results(list(flying), client):
                            flying.discard(future)
                            interval_id, chunk_id, ivpf = fut_meta.pop(
                                future, (None, None, None)
                            )
                            bump()
                            if not ok and interval_id is not None:
                                fail_streak += 1
                                if fail_streak >= fail_threshold:
                                    _trip_to_cpu_only()
                                if retries_left.get((interval_id, chunk_id), 0) > 0:
                                    retries_left[(interval_id, chunk_id)] -= 1
                                    _submit(ivpf, interval_id, chunk_id)
                            else:
                                fail_streak = 0

            for future, ok in _yield_futures_with_results(list(flying), client):
                interval_id, chunk_id, _ = fut_meta.pop(future, (None, None, None))
                bump()
                if not ok and interval_id is not None:
                    logger.error(
                        "GAVE UP after retries | iv %d | chunk %d",
                        interval_id,
                        chunk_id,
                    )

    logger.info("Stage-2 finished – %d tasks submitted", submitted)

