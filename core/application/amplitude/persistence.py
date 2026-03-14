from __future__ import annotations

import logging

import numpy as np

from core.infrastructure.persistence.database_manager import (
    create_db_manager_for_thread,
)
from core.application.amplitude.grid import IntervalTask
from core.application.amplitude.runtime import TIMER, chunk_mutex


logger = logging.getLogger(__name__)


def save_amplitudes_and_meta(
    *,
    chunk_id: int,
    task: IntervalTask,
    grid_shape_nd: np.ndarray,
    total_reciprocal_points: int,
    amplitudes_delta: np.ndarray,
    amplitudes_average: np.ndarray,
    point_data_processor,
    db_path: str,
    quiet_logs: bool = False,
) -> None:
    t0 = TIMER()
    lock = chunk_mutex(chunk_id)

    fn_amp = point_data_processor.data_saver.generate_filename(chunk_id, "_amplitudes")
    fn_amp_av = point_data_processor.data_saver.generate_filename(chunk_id, "_amplitudes_av")
    fn_shape = point_data_processor.data_saver.generate_filename(chunk_id, "_shapeNd")
    fn_tot = point_data_processor.data_saver.generate_filename(
        chunk_id, "_amplitudes_ntotal_reciprocal_space_points"
    )
    fn_nrec = point_data_processor.data_saver.generate_filename(
        chunk_id, "_amplitudes_nreciprocal_space_points"
    )
    fn_applied = point_data_processor.data_saver.generate_filename(
        chunk_id, "_applied_interval_ids"
    )

    already_applied = False

    with lock:
        try:
            point_data_processor.data_saver.load_data(fn_shape)
        except FileNotFoundError:
            point_data_processor.data_saver.save_data({"shapeNd": grid_shape_nd}, fn_shape)

        val = int(total_reciprocal_points)
        try:
            data = point_data_processor.data_saver.load_data(fn_tot)

            def _needs_update(store: dict, key: str) -> bool:
                arr = store.get(key, None)
                if arr is None:
                    return True
                try:
                    v = int(np.array(arr).ravel()[0])
                except Exception:
                    return True
                return v == -1

            if _needs_update(data, "ntotal_reciprocal_space_points") or _needs_update(
                data, "ntotal_reciprocal_points"
            ):
                data["ntotal_reciprocal_space_points"] = np.array([val], dtype=np.int64)
                data["ntotal_reciprocal_points"] = np.array([val], dtype=np.int64)
                point_data_processor.data_saver.save_data(data, fn_tot)
        except FileNotFoundError:
            point_data_processor.data_saver.save_data(
                {
                    "ntotal_reciprocal_space_points": np.array([val], dtype=np.int64),
                    "ntotal_reciprocal_points": np.array([val], dtype=np.int64),
                },
                fn_tot,
            )

        try:
            applied_arr = point_data_processor.data_saver.load_data(fn_applied)["ids"]
            applied_set = set(int(x) for x in np.array(applied_arr).ravel().tolist())
        except FileNotFoundError:
            applied_set = set()

        if int(task.irecip_id) in applied_set:
            already_applied = True
        else:
            try:
                current = point_data_processor.data_saver.load_data(fn_amp)["amplitudes"]
                current_av = point_data_processor.data_saver.load_data(fn_amp_av)["amplitudes_av"]
                nrec = point_data_processor.data_saver.load_data(fn_nrec)["nreciprocal_space_points"]
            except FileNotFoundError:
                current, current_av, nrec = None, None, 0

            if current is None:
                current = amplitudes_delta
                nrec = task.q_grid.shape[0]
            else:
                if task.q_grid.shape[1] > 2 and np.max(np.abs(task.q_grid[:, 2])) > 1e-7:
                    current[:, 1] += amplitudes_delta + np.conj(amplitudes_delta)
                    nrec += task.q_grid.shape[0] * 2
                else:
                    current[:, 1] += amplitudes_delta
                    nrec += task.q_grid.shape[0]

            if current_av is None:
                current_av = amplitudes_average
            else:
                if task.q_grid.shape[1] > 2 and np.max(np.abs(task.q_grid[:, 2])) > 1e-7:
                    current_av[:, 1] += amplitudes_average + np.conj(amplitudes_average)
                else:
                    current_av[:, 1] += amplitudes_average

            point_data_processor._save_chunk_data(chunk_id, None, current, current_av, nrec)
            applied_set.add(int(task.irecip_id))
            applied_sorted = np.array(sorted(applied_set), dtype=np.int64)
            point_data_processor.data_saver.save_data({"ids": applied_sorted}, fn_applied)

    db = create_db_manager_for_thread(db_path)
    try:
        db.update_interval_chunk_status(task.irecip_id, chunk_id, saved=True)
    finally:
        db.close()

    if quiet_logs:
        logger.debug(
            "write-HDF5 | chunk %d | iv %d %s | %.3f s",
            chunk_id,
            task.irecip_id,
            "already applied (idempotent skip)" if already_applied else "applied",
            TIMER() - t0,
        )
    else:
        if already_applied:
            logger.info(
                "write-HDF5 | chunk %d | iv %d already applied (idempotent skip) | %.3f s",
                chunk_id,
                task.irecip_id,
                TIMER() - t0,
            )
        else:
            logger.info(
                "write-HDF5 | chunk %d | iv %d applied | %.3f s",
                chunk_id,
                task.irecip_id,
                TIMER() - t0,
            )
