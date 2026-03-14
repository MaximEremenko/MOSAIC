from __future__ import annotations

import logging
import os

import numpy as np

from core.decoding.decoder_service import (
    apply_decoder,
    build_feature_sets,
    ensure_decoder,
)
from core.decoding.grid import compute_hkl_max_from_intervals
from core.decoding.io import write_displacements_csv
from core.residual_field.loader import (
    load_chunk_residual_field_and_grid,
    resolve_output_dir,
)


def compute_and_save_displacements(
    processor,
    *,
    chunk_id,
    rifft_saver,
    point_data_list,
    output_dir=None,
    broadcast_into_rows=False,
):
    log = logging.getLogger(__name__)
    output_dir = resolve_output_dir(rifft_saver, chunk_id, output_dir)
    data, amplitudes, rifft_space_grid = load_chunk_residual_field_and_grid(
        processor,
        chunk_id=chunk_id,
        point_data_list=point_data_list,
        rifft_saver=rifft_saver,
        logger=log,
    )

    if amplitudes.ndim == 2 and amplitudes.shape[1] >= 2:
        Rvals_all = amplitudes[:, 1]
    else:
        Rvals_all = np.ravel(amplitudes)

    D_all = rifft_space_grid.shape[1] - 1
    coords_all = rifft_space_grid[:, :D_all]
    ids_all = rifft_space_grid[:, -1].astype(int)

    weight_g = float(processor.parameters.get("ls_weight_gamma", 0.35))
    lam_reg = float(processor.parameters.get("dog_lambda_reg", 1e-3))
    max_train = processor.parameters.get("linear_max_training_samples", None)
    intervals = processor.parameters["reciprocal_space_intervals_all"]
    hkl_max_xyz = compute_hkl_max_from_intervals(intervals)
    guard_frac = float(processor.parameters.get("edge_guard_frac", 0.10))
    q_window_kind = str(processor.parameters.get("q_window_kind", "cheb")).lower()
    q_window_at_db = float(processor.parameters.get("q_window_at_db", 100.0))
    size_aver = np.asarray(processor.parameters["supercell"], dtype=int)

    original_coords = processor.original_coords
    average_coords = processor.average_coords
    V = np.asarray(processor.parameters.get("vectors", np.eye(3)), float)
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError(f"parameters['vectors'] must be square; got shape {V.shape}")
    D_disp = int(min(V.shape[0], original_coords.shape[1], average_coords.shape[1]))
    if D_disp <= 0:
        raise ValueError("Could not determine displacement dimensionality (D_disp).")
    Vd = V[:D_disp, :D_disp]
    Vd_inv = np.linalg.inv(Vd)
    decoder_cache_path = processor._get_decoder_cache_path(output_dir)

    features_all, cids_all, features_train, u_train = build_feature_sets(
        processor,
        point_data_list=point_data_list,
        coords_all=coords_all,
        ids_all=ids_all,
        Rvals_all=Rvals_all,
        hkl_max_xyz=hkl_max_xyz,
        q_window_kind=q_window_kind,
        q_window_at_db=q_window_at_db,
        size_aver=size_aver,
        guard_frac=guard_frac,
        original_coords=original_coords,
        average_coords=average_coords,
        Vd_inv=Vd_inv,
        Vd=Vd,
        D_disp=D_disp,
        weight_g=weight_g,
        max_train=max_train,
    )

    ensure_decoder(
        processor,
        cache_path=decoder_cache_path,
        chunk_id=chunk_id,
        features_all=features_all,
        features_train=features_train,
        u_train=u_train,
        lam_reg=lam_reg,
        logger=log,
    )
    U_all = apply_decoder(processor, features_all)

    ids = np.array(cids_all, dtype=np.int64)
    U = U_all.astype(np.float64, copy=False)
    out_table = {
        "central_point_id": ids,
        "u": U,
        "columns": np.array(["ux", "uy", "uz"][: U.shape[1]], dtype=object),
        "coordinate_system": np.array(["cartesian"], dtype=object),
        "units": np.array(["angstrom", "angstrom", "angstrom"][: U.shape[1]], dtype=object),
    }

    h5_path = os.path.join(output_dir, f"output_chunk_{chunk_id}_first_moment_displacements.h5")
    csv_path = os.path.join(output_dir, f"output_chunk_{chunk_id}_first_moment_displacements.csv")
    rifft_saver.save_data(out_table, h5_path)
    write_displacements_csv(csv_path, ids, U)

    if broadcast_into_rows:
        cid2u = {int(i): U[k, :] for k, i in enumerate(ids)}
        urows = np.stack([cid2u[int(c)] for c in ids_all], axis=0)
        if amplitudes.ndim == 1:
            aug = np.column_stack([amplitudes, urows])
        else:
            aug = np.concatenate([amplitudes, urows], axis=1)

        d_aug = dict(data)
        d_aug["amplitudes_with_displacement"] = aug
        d_aug["amplitudes_with_displacement_columns"] = np.array(
            ["<orig...>", *["ux", "uy", "uz"][: urows.shape[1]]], dtype=object
        )
        h5_aug = os.path.join(
            output_dir,
            f"output_chunk_{chunk_id}_amplitudes_with_displacements.h5",
        )
        rifft_saver.save_data(d_aug, h5_aug)

    return out_table
