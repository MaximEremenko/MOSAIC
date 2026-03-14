from __future__ import annotations

import numpy as np

from core.application.postprocessing.decoder_cache import (
    load_decoder_cache,
    save_decoder_cache,
)
from core.application.postprocessing.features import build_feature_vector_from_patch
from core.application.postprocessing.grid import (
    apply_rq_pipeline_local,
    center_patch_subvoxel,
    regrid_patch_to_c,
)


def build_feature_sets(
    processor,
    *,
    point_data_list,
    coords_all,
    ids_all,
    Rvals_all,
    hkl_max_xyz,
    q_window_kind,
    q_window_at_db,
    size_aver,
    guard_frac,
    original_coords,
    average_coords,
    Vd_inv,
    Vd,
    D_disp,
    weight_g,
    max_train,
):
    id2center = {}
    D_all = coords_all.shape[1]
    for point_data in point_data_list:
        cid = int(point_data["central_point_id"])
        id2center[cid] = np.asarray(point_data["coordinates"], float)[:D_all]

    groups = {}
    for index, cid in enumerate(ids_all):
        groups.setdefault(int(cid), []).append(index)

    features_all = []
    cids_all = []
    features_train = []
    u_train = []

    for point_data in point_data_list:
        cid = int(point_data["central_point_id"])
        if cid not in groups:
            continue
        center = id2center.get(cid, None)
        if center is None:
            continue

        idxs = np.asarray(groups[cid], int)
        coords = coords_all[idxs, :]
        Rvals = Rvals_all[idxs]

        Rvals_proc = apply_rq_pipeline_local(
            Rvals,
            coords,
            q_window_kind=q_window_kind,
            q_window_at_db=q_window_at_db,
            size_aver=size_aver,
            hkl_max_xyz=hkl_max_xyz,
            guard_frac=guard_frac,
        )
        y_grid, shape, axes_vals, _ = regrid_patch_to_c(coords, Rvals_proc)
        D = len(shape)
        feat = build_feature_vector_from_patch(
            y_grid,
            axes_vals,
            center_abs=center,
            D=D,
            weight_gamma=weight_g,
            remove_odd_tilt=True,
            center_patch_subvoxel=center_patch_subvoxel,
        )

        features_all.append(feat)
        cids_all.append(cid)

        if processor._decoder_M is None:
            if (max_train is not None) and (len(features_train) >= max_train):
                continue
            if cid < 0 or cid >= original_coords.shape[0]:
                raise IndexError(
                    f"central_point_id {cid} out of bounds for original_coords shape {original_coords.shape}"
                )
            if processor.u_true_all is not None:
                u_true = processor.u_true_all[cid, :D_disp]
            else:
                u_true = (
                    original_coords[cid, :D_disp] @ Vd_inv
                    - average_coords[cid, :D_disp] @ Vd_inv
                )
                u_true = (u_true - np.rint(u_true)) @ Vd
            features_train.append(feat)
            u_train.append(np.asarray(u_true, float))

    return features_all, cids_all, features_train, u_train


def ensure_decoder(
    processor,
    *,
    cache_path,
    chunk_id,
    features_all,
    features_train,
    u_train,
    lam_reg,
    logger,
):
    if processor._decoder_M is None:
        processor._decoder_M, processor._feature_dim = load_decoder_cache(cache_path, logger)

    if not features_all:
        raise RuntimeError("No site features constructed; nothing to do.")

    if processor._decoder_M is None:
        if not features_train:
            raise RuntimeError(
                "Decoder M not present and no training samples collected. "
                "Check that original_coords / average_coords have matching indices with central_point_id."
            )

        R_data = np.stack(features_train, axis=1)
        U_data = np.stack(u_train, axis=1)
        P, N = R_data.shape
        logger.info(
            "Training linear decoder M on chunk %s with %d samples (P=%d).",
            chunk_id,
            N,
            P,
        )
        RR = R_data @ R_data.T
        UR = U_data @ R_data.T
        H = RR + float(lam_reg) * np.eye(P)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            logger.warning("Decoder normal matrix H is singular; using pseudo-inverse.")
            H_inv = np.linalg.pinv(H, rcond=1e-12)
        M = UR @ H_inv
        processor._decoder_M = M.astype(np.float64, copy=False)
        processor._feature_dim = P
        logger.info("Decoder M trained (shape %s).", processor._decoder_M.shape)
        save_decoder_cache(cache_path, processor._decoder_M, processor._feature_dim, logger)

    if processor._feature_dim is None:
        processor._feature_dim = processor._decoder_M.shape[1]
    if any(f.size != processor._feature_dim for f in features_all):
        raise RuntimeError(
            "Feature dimension mismatch: decoder expects "
            f"{processor._feature_dim}, but some features differ."
        )


def apply_decoder(processor, features_all):
    R_all = np.stack(features_all, axis=1)
    return (processor._decoder_M @ R_all).T
