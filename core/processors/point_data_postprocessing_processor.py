# -*- coding: utf-8 -*-
"""
processors/point_data_postprocessing_processor.py
"""

import logging
import os
from collections import defaultdict

import numpy as np
from numba import set_num_threads
from scipy.fft import fft, fftshift

from core.processors.postprocessing_decoder import (
    build_decoder_cache_path,
    load_decoder_cache,
    save_decoder_cache,
)
from core.processors.postprocessing_features import build_feature_vector_from_patch
from core.processors.postprocessing_grid import (
    apply_rq_pipeline_local,
    center_patch_subvoxel,
    compute_hkl_max_from_intervals,
    regrid_patch_to_c,
)
from core.processors.postprocessing_io import (
    normalize_amplitudes_ntotal,
    write_displacements_csv,
    write_site_intensities_csv,
)

set_num_threads(32)


class PointDataPostprocessingProcessor:
    def __init__(self, db_manager, point_data_processor, parameters):
        self.db_manager = db_manager
        self.point_data_processor = point_data_processor
        self.parameters = dict(parameters or {})

        rspace_info = self.parameters.get("rspace_info") or {}
        mode_raw = (
            self.parameters.get("postprocessing_mode")
            or self.parameters.get("postprocess_mode")
            or self.parameters.get("mode")
            or rspace_info.get("postprocessing_mode")
            or rspace_info.get("postprocess_mode")
            or rspace_info.get("mode")
            or "displacement"
        )
        mode_norm = str(mode_raw).strip().lower()
        if mode_norm in (
            "chemical",
            "chem",
            "checmical",
            "occupational",
            "occupancy",
            "occupantioal",
        ):
            self.mode = "chemical"
        else:
            self.mode = "displacement"

        self.parameters.setdefault("normalize_amplitudes_by", "ntotal")
        self.parameters.setdefault("coords_are_fractional", False)
        self.parameters.setdefault("ls_weight_gamma", 0.35)
        self.parameters.setdefault("dog_lambda_reg", 1e-3)
        self.parameters.setdefault("linear_max_training_samples", None)
        self.parameters.setdefault("q_window_kind", "cheb")
        self.parameters.setdefault("q_window_at_db", 100.0)
        self.parameters.setdefault("edge_guard_frac", 0.10)

        self.original_coords = None
        self.average_coords = None
        if self.mode == "displacement":
            if "original_coords" not in self.parameters:
                raise KeyError("parameters['original_coords'] is required for displacement mode.")
            if "average_coords" not in self.parameters:
                raise KeyError(
                    "parameters['average_coords'] is required for displacement mode. "
                    "Add avg_coords.to_numpy() to params in main.py."
                )
            self.original_coords = np.asarray(self.parameters["original_coords"], float)
            self.average_coords = np.asarray(self.parameters["average_coords"], float)

        self.u_true_all = None
        if "displacements_from_config" in self.parameters:
            self.u_true_all = np.asarray(self.parameters["displacements_from_config"], float)
        self._decoder_M = None
        self._feature_dim = None

    def process_chunk(self, chunk_id, rifft_saver, client, output_dir):
        point_data_list = self.db_manager.get_point_data_for_chunk(chunk_id)
        if not point_data_list:
            print(f"No point data found for chunk_id: {chunk_id}")
            return None

        if self.mode == "chemical":
            return self.compute_and_save_site_intensities(
                chunk_id=chunk_id,
                rifft_saver=rifft_saver,
                point_data_list=point_data_list,
                output_dir=output_dir,
            )

        return self.compute_and_save_displacements(
            chunk_id=chunk_id,
            rifft_saver=rifft_saver,
            point_data_list=point_data_list,
            output_dir=output_dir,
            broadcast_into_rows=self.parameters.get(
                "broadcast_displacement_into_rows", False
            ),
        )

    def _get_decoder_cache_path(self, output_dir: str) -> str:
        return build_decoder_cache_path(self.parameters, output_dir)

    def load_amplitudes_and_generate_grid(self, chunk_id, point_data_list, rifft_saver):
        filename = rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
        try:
            data = rifft_saver.load_data(filename)
            amplitudes = data.get("amplitudes", None)
            if amplitudes is None:
                print(f"Amplitudes not found in {filename}")
                return np.array([]), None, None

            amplitudes = normalize_amplitudes_ntotal(
                amplitudes,
                rifft_saver=rifft_saver,
                chunk_id=chunk_id,
                logger=logging.getLogger(__name__),
            )

            grids = []
            grids_shapeNd = []
            central_point_ids = []
            for point_data in point_data_list:
                grid_points, grid_shapeNd = self.point_data_processor.generate_grid(
                    chunk_id=chunk_id,
                    dimensionality=len(point_data["coordinates"]),
                    step_in_frac=point_data["step_in_frac"],
                    central_point=point_data["coordinates"],
                    dist=point_data["dist_from_atom_center"],
                    central_point_id=point_data["central_point_id"],
                )
                grids.append(grid_points)
                grids_shapeNd.append(grid_shapeNd)
                central_point_ids.extend(
                    [point_data["central_point_id"]] * len(grid_points)
                )

            rifft_space_grid = (
                np.hstack((np.vstack(grids), np.array(central_point_ids)[:, None]))
                if grids
                else np.array([])
            )
            return rifft_space_grid, amplitudes, grids_shapeNd
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return np.array([]), None, None

    def compute_and_save_displacements(
        self,
        *,
        chunk_id,
        rifft_saver,
        point_data_list,
        output_dir=None,
        broadcast_into_rows=False,
    ):
        log = logging.getLogger(__name__)

        if output_dir is None:
            out_by_saver = getattr(rifft_saver, "output_dir", None)
            output_dir = out_by_saver or os.path.dirname(
                os.path.abspath(
                    rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
                )
            )
        os.makedirs(output_dir, exist_ok=True)

        fn_amp = rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
        try:
            d = rifft_saver.load_data(fn_amp)
            amplitudes = d.get("amplitudes", None)
            rifft_space_grid = d.get("rifft_space_grid", None)
        except FileNotFoundError:
            d = {}
            amplitudes = None
            rifft_space_grid = None

        if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid) == 0:
            rifft_space_grid2, amplitudes2, _ = self.load_amplitudes_and_generate_grid(
                chunk_id, point_data_list, rifft_saver
            )
            if amplitudes is None:
                amplitudes = amplitudes2
            if rifft_space_grid is None:
                rifft_space_grid = rifft_space_grid2
            if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid) == 0:
                raise RuntimeError(f"Nothing to process for chunk {chunk_id}")
            amplitudes = normalize_amplitudes_ntotal(
                amplitudes,
                rifft_saver=rifft_saver,
                chunk_id=chunk_id,
                logger=log,
            )

        if amplitudes.ndim == 2 and amplitudes.shape[1] >= 2:
            Rvals_all = amplitudes[:, 1]
        else:
            Rvals_all = np.ravel(amplitudes)

        rifft_space_grid = np.asarray(rifft_space_grid)
        D_all = rifft_space_grid.shape[1] - 1
        coords_all = rifft_space_grid[:, :D_all]
        ids_all = rifft_space_grid[:, -1].astype(int)

        id2center = {}
        for point_data in point_data_list:
            cid = int(point_data["central_point_id"])
            id2center[cid] = np.asarray(point_data["coordinates"], float)[:D_all]

        groups = defaultdict(list)
        for index, cid in enumerate(ids_all):
            groups[int(cid)].append(index)

        weight_g = float(self.parameters.get("ls_weight_gamma", 0.35))
        lam_reg = float(self.parameters.get("dog_lambda_reg", 1e-3))
        max_train = self.parameters.get("linear_max_training_samples", None)

        intervals = self.parameters["reciprocal_space_intervals_all"]
        hkl_max_xyz = compute_hkl_max_from_intervals(intervals)
        guard_frac = float(self.parameters.get("edge_guard_frac", 0.10))
        q_window_kind = str(self.parameters.get("q_window_kind", "cheb")).lower()
        q_window_at_db = float(self.parameters.get("q_window_at_db", 100.0))
        size_aver = np.asarray(self.parameters["supercell"], dtype=int)

        original_coords = self.original_coords
        average_coords = self.average_coords
        V = np.asarray(self.parameters.get("vectors", np.eye(3)), float)
        if V.ndim != 2 or V.shape[0] != V.shape[1]:
            raise ValueError(f"parameters['vectors'] must be square; got shape {V.shape}")
        D_disp = int(min(V.shape[0], original_coords.shape[1], average_coords.shape[1]))
        if D_disp <= 0:
            raise ValueError("Could not determine displacement dimensionality (D_disp).")
        Vd = V[:D_disp, :D_disp]
        Vd_inv = np.linalg.inv(Vd)

        decoder_cache_path = self._get_decoder_cache_path(output_dir)
        if self._decoder_M is None:
            self._decoder_M, self._feature_dim = load_decoder_cache(
                decoder_cache_path, log
            )

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

            if self._decoder_M is None:
                if (max_train is not None) and (len(features_train) >= max_train):
                    continue
                if cid < 0 or cid >= original_coords.shape[0]:
                    raise IndexError(
                        f"central_point_id {cid} out of bounds for original_coords shape {original_coords.shape}"
                    )

                if self.u_true_all is not None:
                    u_true = self.u_true_all[cid, :D_disp]
                else:
                    u_true = (
                        original_coords[cid, :D_disp] @ Vd_inv
                        - average_coords[cid, :D_disp] @ Vd_inv
                    )
                    u_true = (u_true - np.rint(u_true)) @ Vd

                features_train.append(feat)
                u_train.append(np.asarray(u_true, float))

        if not features_all:
            raise RuntimeError("No site features constructed; nothing to do.")

        if self._decoder_M is None:
            if not features_train:
                raise RuntimeError(
                    "Decoder M not present and no training samples collected. "
                    "Check that original_coords / average_coords have matching "
                    "indices with central_point_id."
                )

            R_data = np.stack(features_train, axis=1)
            U_data = np.stack(u_train, axis=1)
            P, N = R_data.shape
            log.info(
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
                log.warning("Decoder normal matrix H is singular; using pseudo-inverse.")
                H_inv = np.linalg.pinv(H, rcond=1e-12)
            M = UR @ H_inv
            self._decoder_M = M.astype(np.float64, copy=False)
            self._feature_dim = P
            log.info("Decoder M trained (shape %s).", self._decoder_M.shape)
            save_decoder_cache(
                decoder_cache_path,
                self._decoder_M,
                self._feature_dim,
                log,
            )

        if self._feature_dim is None:
            self._feature_dim = self._decoder_M.shape[1]
        if any(f.size != self._feature_dim for f in features_all):
            raise RuntimeError(
                "Feature dimension mismatch: decoder expects "
                f"{self._feature_dim}, but some features differ."
            )

        R_all = np.stack(features_all, axis=1)
        U_all = (self._decoder_M @ R_all).T

        ids = np.array(cids_all, dtype=np.int64)
        U = U_all.astype(np.float64, copy=False)
        out_table = {
            "central_point_id": ids,
            "u": U,
            "columns": np.array(["ux", "uy", "uz"][: U.shape[1]], dtype=object),
            "coordinate_system": np.array(["cartesian"], dtype=object),
            "units": np.array(
                ["angstrom", "angstrom", "angstrom"][: U.shape[1]], dtype=object
            ),
        }

        h5_path = os.path.join(
            output_dir, f"output_chunk_{chunk_id}_first_moment_displacements.h5"
        )
        csv_path = os.path.join(
            output_dir, f"output_chunk_{chunk_id}_first_moment_displacements.csv"
        )
        rifft_saver.save_data(out_table, h5_path)
        write_displacements_csv(csv_path, ids, U)

        if broadcast_into_rows:
            cid2u = {int(i): U[k, :] for k, i in enumerate(ids)}
            urows = np.stack([cid2u[int(c)] for c in ids_all], axis=0)
            if amplitudes.ndim == 1:
                aug = np.column_stack([amplitudes, urows])
            else:
                aug = np.concatenate([amplitudes, urows], axis=1)

            d_aug = dict(d)
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

    def compute_and_save_site_intensities(
        self,
        *,
        chunk_id,
        rifft_saver,
        point_data_list,
        output_dir=None,
    ):
        log = logging.getLogger(__name__)

        if output_dir is None:
            out_by_saver = getattr(rifft_saver, "output_dir", None)
            output_dir = out_by_saver or os.path.dirname(
                os.path.abspath(
                    rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
                )
            )
        os.makedirs(output_dir, exist_ok=True)

        fn_amp = rifft_saver.generate_filename(chunk_id, suffix="_amplitudes")
        try:
            d = rifft_saver.load_data(fn_amp)
            amplitudes = d.get("amplitudes", None)
            rifft_space_grid = d.get("rifft_space_grid", None)
        except FileNotFoundError:
            d = {}
            amplitudes = None
            rifft_space_grid = None

        if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid) == 0:
            rifft_space_grid2, amplitudes2, _ = self.load_amplitudes_and_generate_grid(
                chunk_id, point_data_list, rifft_saver
            )
            if amplitudes is None:
                amplitudes = amplitudes2
            if rifft_space_grid is None:
                rifft_space_grid = rifft_space_grid2
            if amplitudes is None or rifft_space_grid is None or len(rifft_space_grid) == 0:
                raise RuntimeError(f"Nothing to process for chunk {chunk_id}")

            amplitudes = normalize_amplitudes_ntotal(
                amplitudes,
                rifft_saver=rifft_saver,
                chunk_id=chunk_id,
                logger=log,
            )

        if amplitudes.ndim == 2 and amplitudes.shape[1] >= 2:
            Rvals_all = amplitudes[:, 1]
        else:
            Rvals_all = np.ravel(amplitudes)

        rifft_space_grid = np.asarray(rifft_space_grid)
        D_all = rifft_space_grid.shape[1] - 1
        coords_all = rifft_space_grid[:, :D_all]
        ids_all = rifft_space_grid[:, -1].astype(int)

        groups = defaultdict(list)
        for index, cid in enumerate(ids_all):
            groups[int(cid)].append(index)

        elements_all = self.parameters.get("elements", None)
        refnumbers_all = self.parameters.get("refnumbers", None)
        elements_all = None if elements_all is None else np.asarray(elements_all)
        refnumbers_all = None if refnumbers_all is None else np.asarray(refnumbers_all)

        out_ids = []
        out_coords = []
        out_real = []
        out_imag = []

        for point_data in point_data_list:
            cid = int(point_data["central_point_id"])
            idxs = groups.get(cid, None)
            if not idxs:
                continue
            idxs = np.asarray(idxs, dtype=int)

            center = np.asarray(point_data["coordinates"], float)[:D_all]
            coords = coords_all[idxs, :]

            is_center = np.all(
                np.isclose(coords, center[None, :], atol=1e-10, rtol=0.0),
                axis=1,
            )
            if np.any(is_center):
                pick = int(np.flatnonzero(is_center)[0])
            else:
                pick = int(np.argmin(np.linalg.norm(coords - center[None, :], axis=1)))

            val = complex(Rvals_all[int(idxs[pick])])
            out_ids.append(cid)
            out_coords.append(center)
            out_real.append(float(np.real(val)))
            out_imag.append(float(np.imag(val)))

        if not out_ids:
            raise RuntimeError("No site intensities extracted; nothing to do.")

        order = np.argsort(np.asarray(out_ids, dtype=np.int64))
        ids = np.asarray(out_ids, dtype=np.int64)[order]
        coords = np.asarray(out_coords, float)[order]
        intens_real = np.asarray(out_real, float)[order]
        intens_imag = np.asarray(out_imag, float)[order]

        out_table = {
            "central_point_id": ids,
            "coordinates": coords,
            "intensity_real": intens_real,
            "intensity_imag": intens_imag,
        }

        if (
            elements_all is not None
            and elements_all.ndim == 1
            and elements_all.shape[0] > int(ids.max())
        ):
            out_table["element"] = np.asarray(elements_all[ids], dtype=object)
        if (
            refnumbers_all is not None
            and refnumbers_all.ndim == 1
            and refnumbers_all.shape[0] > int(ids.max())
        ):
            out_table["refnumber"] = np.asarray(refnumbers_all[ids])

        h5_path = os.path.join(
            output_dir, f"output_chunk_{chunk_id}_chemical_site_intensities.h5"
        )
        csv_path = os.path.join(
            output_dir, f"output_chunk_{chunk_id}_chemical_site_intensities.csv"
        )
        rifft_saver.save_data(out_table, h5_path)
        write_site_intensities_csv(
            csv_path,
            ids,
            coords,
            intens_real,
            intens_imag,
            elements=out_table.get("element", None),
            refnumbers=out_table.get("refnumber", None),
        )
        return out_table

    def filter_from_window(
        self, window0, dimensionality, window1=None, window2=None, size_aver=None
    ):
        if dimensionality == 1:
            fd0 = fft(window0, np.array(self.parameters["supercell"])[0]) / (
                len(window0) / 2.0
            )
            k0 = np.abs(fftshift(fd0 / np.abs(fd0).max()))
            return k0 / k0.sum()
        if dimensionality == 2:
            fd0 = fft(window0, self.parameters["supercell"][0]) / (
                len(window0) / 2.0
            )
            fd1 = fft(window1, self.parameters["supercell"][1]) / (
                len(window1) / 2.0
            )
            k0 = np.abs(fftshift(fd0 / np.abs(fd0).max()))
            k1 = np.abs(fftshift(fd1 / np.abs(fd1).max()))
            kern = k0[:, None] * k1[None, :]
            return kern / kern.sum()
        if dimensionality == 3:
            sc = np.array(self.parameters["supercell"])
            fd0 = fft(window0, sc[0]) / (len(window0) / 2.0)
            fd1 = fft(window1, sc[1]) / (len(window1) / 2.0)
            fd2 = fft(window2, sc[2]) / (len(window2) / 2.0)
            k0 = np.abs(fftshift(fd0 / np.abs(fd0).max()))
            k1 = np.abs(fftshift(fd1 / np.abs(fd1).max()))
            k2 = np.abs(fftshift(fd2 / np.abs(fd2).max()))
            kern = k0[:, None, None] * k1[None, :, None] * k2[None, None, :]
            return kern / kern.sum()
        raise ValueError("Unsupported dimensionality")
