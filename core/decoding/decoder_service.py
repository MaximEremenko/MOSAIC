from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from core.decoding.context import build_decoding_context
from core.decoding.contracts import (
    DecoderSourceProvenance,
    DisplacementDecoderSourcePolicy,
)
from core.decoding.decoder_cache import (
    build_decoder_cache_path,
    load_decoder_cache,
    save_decoder_provenance,
    save_decoder_cache,
)
from core.decoding.payloads import build_decoding_payload
from core.decoding.features import build_feature_vector_from_patch
from core.decoding.grid import (
    apply_rq_pipeline_local,
    center_patch_subvoxel,
    regrid_patch_to_c,
)
from core.decoding.loader import resolve_output_dir
from core.decoding.state import build_postprocessing_processor_state
from core.patch_centers.contracts import PointSelectionRequest


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


def build_decoder_training_payload(
    processor,
    *,
    chunk_id,
    rifft_saver,
    point_data_list,
    output_dir=None,
):
    output_dir = resolve_output_dir(rifft_saver, chunk_id, output_dir)
    from core.decoding.displacement_service import prepare_displacement_decoder_inputs

    return prepare_displacement_decoder_inputs(
        processor,
        chunk_id=chunk_id,
        rifft_saver=rifft_saver,
        point_data_list=point_data_list,
        output_dir=output_dir,
    )


def load_required_decoder(cache_path: str, logger):
    decoder, feature_dim = load_decoder_cache(cache_path, logger)
    if decoder is None or feature_dim is None:
        raise FileNotFoundError(
            "No usable M-decoder was found at "
            f"'{cache_path}'. Provide a valid processing.decoder.cache_path or "
            "use processing.decoder.source='compute'."
        )
    return decoder, feature_dim


def train_decoder_from_samples(
    processor,
    *,
    cache_path: str,
    training_features: list[np.ndarray],
    training_targets: list[np.ndarray],
    lam_reg: float,
    logger,
    label: str,
) -> None:
    if not training_features:
        raise RuntimeError(
            "No displacement decoder training samples were collected for "
            f"{label}. A full/unmasked decoder source run is required."
        )
    R_data = np.stack(training_features, axis=1)
    U_data = np.stack(training_targets, axis=1)
    P, N = R_data.shape
    logger.info(
        "Training linear decoder M for %s with %d samples (P=%d).",
        label,
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


def ensure_decoder(
    processor,
    *,
    features_all,
    logger=None,
):
    if not features_all:
        raise RuntimeError("No site features constructed; nothing to do.")
    if processor._decoder_M is None or processor._feature_dim is None:
        raise RuntimeError(
            "No prepared M-decoder is available for displacement decoding. "
            "Set processing.decoder.source to 'cache' with a valid cache_path, "
            "or use processing.decoder.source='compute' with a separate "
            "compute_output_directory."
        )

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


class DisplacementDecoderSourceService:
    def __init__(
        self,
        *,
        point_selection_service,
        reciprocal_space_service,
        scattering_stage,
        residual_field_stage,
    ) -> None:
        self.point_selection_service = point_selection_service
        self.reciprocal_space_service = reciprocal_space_service
        self.scattering_stage = scattering_stage
        self.residual_field_stage = residual_field_stage

    def prepare(
        self,
        *,
        processor,
        workflow_parameters,
        structure,
        artifacts,
        client,
    ) -> DecoderSourceProvenance | None:
        state = build_postprocessing_processor_state(processor.parameters)
        if state.mode != "displacement":
            return None
        policy = state.decoder_source_policy or DisplacementDecoderSourcePolicy()
        logger = logging.getLogger(__name__)
        output_dir = str(artifacts.output_dir)
        processor.decoder_source_policy = policy

        if policy.mode == "error":
            if processor._decoder_M is not None and processor._feature_dim is not None:
                provenance = DecoderSourceProvenance(
                    mode="error",
                    semantics="preloaded",
                    decoder_cache_path="<in-memory>",
                    feature_dim=processor._feature_dim,
                    loaded_from_cache=False,
                    computed=False,
                )
                processor.decoder_source_provenance = provenance
                save_decoder_provenance(output_dir, provenance.to_mapping(), logger)
                return provenance
            raise RuntimeError(
                "Displacement decoding now requires an explicit decoder source. "
                "Set processing.decoder.source to 'cache' with a valid cache_path, "
                "or to 'compute' with a separate compute_output_directory. "
                "No implicit M-decoder training is performed; this is especially "
                "important for expensive 3D runs."
            )

        if policy.mode == "cache":
            decoder, feature_dim = load_required_decoder(policy.cache_path, logger)
            processor._decoder_M = decoder
            processor._feature_dim = feature_dim
            provenance = DecoderSourceProvenance(
                mode="cache",
                semantics="precomputed",
                decoder_cache_path=str(policy.cache_path),
                source_output_directory=str(Path(policy.cache_path).resolve().parent),
                feature_dim=feature_dim,
                loaded_from_cache=True,
                computed=False,
            )
            processor.decoder_source_provenance = provenance
            save_decoder_provenance(output_dir, provenance.to_mapping(), logger)
            return provenance

        cache_path, provenance = self._compute_decoder_cache(
            processor=processor,
            policy=policy,
            workflow_parameters=workflow_parameters,
            structure=structure,
            client=client,
        )
        decoder, feature_dim = load_required_decoder(cache_path, logger)
        processor._decoder_M = decoder
        processor._feature_dim = feature_dim
        finalized = DecoderSourceProvenance(
            mode="compute",
            semantics="unmasked",
            decoder_cache_path=str(cache_path),
            source_output_directory=str(Path(cache_path).resolve().parent),
            compute_output_directory=provenance.compute_output_directory,
            feature_dim=feature_dim,
            loaded_from_cache=provenance.loaded_from_cache,
            computed=provenance.computed,
        )
        processor.decoder_source_provenance = finalized
        save_decoder_provenance(output_dir, finalized.to_mapping(), logger)
        return finalized

    def _compute_decoder_cache(
        self,
        *,
        processor,
        policy: DisplacementDecoderSourcePolicy,
        workflow_parameters,
        structure,
        client,
    ) -> tuple[str, DecoderSourceProvenance]:
        logger = logging.getLogger(__name__)
        if policy is None or policy.compute_output_directory is None:
            raise RuntimeError(
                "processing.decoder.compute_output_directory is required when "
                "processing.decoder.source='compute'."
            )

        compute_root = Path(policy.compute_output_directory).resolve()
        masked_output_root = Path(workflow_parameters.struct_info.working_directory).resolve()
        if compute_root == masked_output_root:
            raise RuntimeError(
                "processing.decoder.compute_output_directory must differ from the "
                "masked run output directory so masked and unmasked artifacts do not mix."
            )

        compute_params = self._build_unmasked_workflow_parameters(
            workflow_parameters,
            compute_root,
        )
        compute_processed_dir = compute_root / "processed_point_data"
        cache_path = build_decoder_cache_path(
            processor.parameters,
            str(compute_processed_dir),
        )
        if Path(cache_path).is_file():
            decoder, feature_dim = load_required_decoder(cache_path, logger)
            processor._decoder_M = decoder
            processor._feature_dim = feature_dim
            provenance = DecoderSourceProvenance(
                mode="compute",
                semantics="unmasked",
                decoder_cache_path=str(cache_path),
                source_output_directory=str(compute_processed_dir),
                compute_output_directory=str(compute_root),
                feature_dim=feature_dim,
                loaded_from_cache=True,
                computed=False,
            )
            save_decoder_provenance(
                str(compute_processed_dir),
                provenance.to_mapping(),
                logger,
            )
            return cache_path, provenance
        decoding_context = None
        compute_artifacts = None
        try:
            point_data = self.point_selection_service.select(
                PointSelectionRequest(
                    method=compute_params.rspace_info.method,
                    parameters=compute_params,
                    structure=structure,
                    hdf5_file_path=str(compute_processed_dir / "point_data.hdf5"),
                )
            )
            compute_artifacts = self.reciprocal_space_service.prepare(
                workflow_parameters=compute_params,
                point_data=point_data,
                supercell=structure.supercell,
                output_dir=str(compute_processed_dir),
            )
            scattering_parameters = self.scattering_stage.execute(
                workflow_parameters=compute_params,
                structure=structure,
                artifacts=compute_artifacts,
                client=client,
            )
            self.residual_field_stage.execute(
                workflow_parameters=compute_params,
                structure=structure,
                artifacts=compute_artifacts,
                client=client,
                scattering_parameters=scattering_parameters,
            )
            decoding_context = build_decoding_context(
                workflow_parameters=compute_params,
                structure=structure,
                artifacts=compute_artifacts,
            )
            decoding_parameters = {
                **processor.parameters,
                **build_decoding_payload(decoding_context),
            }
            from core.decoding.processor import PointDataPostprocessingProcessor

            training_processor = PointDataPostprocessingProcessor(
                compute_artifacts.db_manager,
                compute_artifacts.point_data_processor,
                decoding_parameters,
            )

            training_features: list[np.ndarray] = []
            training_targets: list[np.ndarray] = []
            max_train = training_processor.parameters.get("linear_max_training_samples")
            for chunk_id in sorted(compute_artifacts.db_manager.get_pending_chunk_ids()):
                point_data_list = compute_artifacts.db_manager.get_point_data_for_chunk(
                    int(chunk_id)
                )
                training_payload = build_decoder_training_payload(
                    training_processor,
                    chunk_id=int(chunk_id),
                    rifft_saver=compute_artifacts.saver,
                    point_data_list=point_data_list,
                    output_dir=str(compute_processed_dir),
                )
                training_features.extend(training_payload["features_train"])
                training_targets.extend(training_payload["u_train"])
            if max_train is not None:
                limit = int(max_train)
                training_features = training_features[:limit]
                training_targets = training_targets[:limit]
            train_decoder_from_samples(
                training_processor,
                cache_path=cache_path,
                training_features=training_features,
                training_targets=training_targets,
                lam_reg=float(training_processor.parameters.get("dog_lambda_reg", 1e-3)),
                logger=logger,
                label="full/unmasked decoder-source run",
            )
            provenance = DecoderSourceProvenance(
                mode="compute",
                semantics="unmasked",
                decoder_cache_path=str(cache_path),
                source_output_directory=str(compute_processed_dir),
                compute_output_directory=str(compute_root),
                feature_dim=training_processor._feature_dim,
                loaded_from_cache=False,
                computed=True,
            )
            save_decoder_provenance(
                str(compute_processed_dir),
                provenance.to_mapping(),
                logger,
            )
            return cache_path, provenance
        finally:
            if compute_artifacts is not None:
                compute_artifacts.close()

    def _build_unmasked_workflow_parameters(self, workflow_parameters, compute_root: Path):
        payload = workflow_parameters.to_payload()
        peak_info = dict(payload.get("peakInfo", {}))
        peak_info.pop("mask_equation", None)
        peak_info.pop("specialPoints", None)
        peak_info.pop("r1", None)
        peak_info.pop("r2", None)
        payload["peakInfo"] = peak_info
        struct_info = dict(payload.get("structInfo", {}))
        struct_info["working_directory"] = str(compute_root)
        payload["structInfo"] = struct_info
        rspace_info = dict(payload.get("rspace_info", {}))
        rspace_info["run_postprocessing"] = False
        rspace_info["decoder"] = {"source": "error"}
        payload["rspace_info"] = rspace_info
        from core.models import WorkflowParameters

        return WorkflowParameters.from_payload(payload)
