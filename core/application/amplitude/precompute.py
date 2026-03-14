from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from dask.distributed import Client
from tqdm.contrib.logging import logging_redirect_tqdm

from core.infrastructure.persistence.database_manager import DatabaseManager
from core.application.amplitude.grid import (
    IntervalTask,
    generate_q_space_grid_sync,
)
from core.application.amplitude.runtime import (
    _is_sync_client,
    _tqdm,
    _yield_futures_with_results,
)
from core.infrastructure.adapters.cunufft_wrapper import execute_cunufft


logger = logging.getLogger(__name__)


def _process_interval_element(
    iv: dict,
    q_grid: np.ndarray,
    element: str,
    orig_coords: np.ndarray,
    cell_orig: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    ff_factory,
) -> Tuple | None:
    ff = ff_factory.calculate(q_grid, element, charge=charge)
    mask = elements_arr == element
    if not np.any(mask):
        return None
    q_amp = ff * execute_cunufft(orig_coords[mask], np.ones(mask.sum()), q_grid, eps=1e-12)
    q_av = execute_cunufft(cell_orig, np.ones(orig_coords.shape[0]), q_grid, eps=1e-12)
    q_del = execute_cunufft(
        orig_coords[mask] - cell_orig[mask], np.ones(mask.sum()), q_grid, eps=1e-12
    )
    q_av_final = ff * q_av * q_del / orig_coords.shape[0]
    return (iv["id"], element, q_grid, q_amp, q_av_final)


def _process_interval_coeff(
    iv: dict,
    q_grid: np.ndarray,
    coeff: np.ndarray,
    orig_coords: np.ndarray,
    cell_orig: np.ndarray,
) -> Tuple:
    n_points = orig_coords.shape[0]
    c_ = coeff * (np.ones(n_points) + 1j * np.zeros(n_points))
    q_amplitudes = execute_cunufft(orig_coords, c_, q_grid, eps=1e-12)
    q_amplitudes_av = execute_cunufft(cell_orig, c_ * 0.0 + 1.0, q_grid, eps=1e-12)
    q_amplitudes_delta = execute_cunufft(
        orig_coords - cell_orig, c_, q_grid, eps=1e-12
    )
    q_amplitudes_av_final = q_amplitudes_av * q_amplitudes_delta / n_points
    return (iv["id"], "All", q_grid, q_amplitudes, q_amplitudes_av_final)


def aggregate_interval_tasks(tasks: List[tuple], use_coeff: bool) -> IntervalTask:
    if use_coeff:
        irecip_id, element, qg, qa, qav = tasks[0]
        return IntervalTask(irecip_id, element, qg, qa, qav)

    irecip_id = tasks[0][0]
    q_grid = tasks[0][2]
    q_amp = np.sum([task[3] for task in tasks], axis=0)
    q_av = np.sum([task[4] for task in tasks], axis=0)
    return IntervalTask(irecip_id, "All", q_grid, q_amp, q_av)


def handle_interval_worker(
    iv: dict,
    *,
    B_: np.ndarray,
    mask_params: dict,
    MaskStrategy,
    supercell: np.ndarray,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    use_coeff: bool,
    coeff_val: np.ndarray | None,
    unique_elements: list[str],
    ff_factory,
    out_dir: str,
    db_path: str,
) -> str | None:
    from core.infrastructure.persistence.database_manager import (
        create_db_manager_for_thread,
    )

    db = create_db_manager_for_thread(db_path)

    if db.is_interval_precomputed(iv["id"]):
        interval_path = Path(out_dir) / f"interval_{iv['id']}.npz"
        if interval_path.exists():
            db.close()
            return str(interval_path)

    q_grid = generate_q_space_grid_sync(iv, B_, mask_params, MaskStrategy, supercell)
    if q_grid.size == 0:
        db.close()
        return None

    tasks: list[tuple] = []
    if use_coeff:
        tasks.append(
            _process_interval_coeff(
                iv, q_grid, coeff_val, original_coords, cells_origin
            )
        )
    else:
        for element in unique_elements:
            task = _process_interval_element(
                iv,
                q_grid,
                element,
                original_coords,
                cells_origin,
                elements_arr,
                charge,
                ff_factory,
            )
            if task is not None:
                tasks.append(task)

    if not tasks:
        db.close()
        return None

    interval_task = aggregate_interval_tasks(tasks, use_coeff)
    out_path = Path(out_dir) / f"interval_{interval_task.irecip_id}.npz"
    with tempfile.NamedTemporaryFile(
        dir=out_dir,
        prefix=f"interval_{interval_task.irecip_id}_",
        suffix=".npz",
        delete=False,
    ) as tf:
        np.savez_compressed(
            tf,
            irecip_id=interval_task.irecip_id,
            element=interval_task.element,
            q_grid=interval_task.q_grid,
            q_amp=interval_task.q_amp,
            q_amp_av=interval_task.q_amp_av,
        )
    Path(tf.name).replace(out_path)

    db.mark_interval_precomputed(interval_task.irecip_id, True)
    db.close()
    return str(out_path)


def precompute_intervals(
    reciprocal_space_intervals: Iterable[dict],
    *,
    B_: np.ndarray,
    parameters: Dict[str, Any],
    unique_elements: Iterable[str],
    mask_params: Dict[str, Any],
    MaskStrategy,
    supercell: np.ndarray,
    out_dir: Path,
    original_coords: np.ndarray,
    cells_origin: np.ndarray,
    elements_arr: np.ndarray,
    charge: float,
    ff_factory,
    db: DatabaseManager,
    client: Client | None,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_coeff = "coeff" in parameters
    coeff_val = parameters.get("coeff")

    todo, cached = [], []
    for interval in reciprocal_space_intervals:
        interval_id = interval["id"]
        path = out_dir / f"interval_{interval_id}.npz"
        if db.is_interval_precomputed(interval_id) and path.exists():
            cached.append(path)
        else:
            todo.append(interval)

    if not todo:
        logger.info("Stage-1 complete: %d written, %d cached, %d skipped", 0, len(cached), 0)
        return cached

    written_files: List[Path] = []
    if (client is not None) and (not _is_sync_client(client)):
        futures = [
            client.submit(
                handle_interval_worker,
                interval,
                B_=B_,
                mask_params=mask_params,
                MaskStrategy=MaskStrategy,
                supercell=supercell,
                original_coords=original_coords,
                cells_origin=cells_origin,
                elements_arr=elements_arr,
                charge=charge,
                use_coeff=use_coeff,
                coeff_val=coeff_val,
                unique_elements=list(unique_elements),
                ff_factory=ff_factory,
                out_dir=str(out_dir),
                db_path=db.db_path,
                pure=False,
                resources={"nufft": 1},
            )
            for interval in todo
        ]
        with logging_redirect_tqdm():
            with _tqdm(len(futures), desc="Precompute intervals", unit="intervals") as pbar:
                for future, _ok in _yield_futures_with_results(futures, client):
                    try:
                        path = future.result()
                    except Exception:
                        path = None
                    if path:
                        written_files.append(Path(path))
                    pbar.update(1)
                    pbar.refresh()
        logger.info(
            "Stage-1 complete: %d written, %d cached, %d skipped",
            len(written_files),
            len(cached),
            len(todo) - len(written_files),
        )
        return cached + written_files

    with _tqdm(len(todo), desc="Precompute intervals", unit="intervals") as pbar:
        for interval in todo:
            path = handle_interval_worker(
                interval,
                B_=B_,
                mask_params=mask_params,
                MaskStrategy=MaskStrategy,
                supercell=supercell,
                original_coords=original_coords,
                cells_origin=cells_origin,
                elements_arr=elements_arr,
                charge=charge,
                use_coeff=use_coeff,
                coeff_val=coeff_val,
                unique_elements=list(unique_elements),
                ff_factory=ff_factory,
                out_dir=str(out_dir),
                db_path=db.db_path,
            )
            if path:
                written_files.append(Path(path))
            pbar.update(1)
            pbar.refresh()

    logger.info(
        "Stage-1 complete: %d written, %d cached, %d skipped",
        len(written_files),
        len(cached),
        len(todo) - len(written_files),
    )
    return cached + written_files
