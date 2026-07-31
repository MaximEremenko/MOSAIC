from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from core.decoding.context import build_decoding_context
from core.decoding.contracts import (
    DecoderSourceProvenance,
    DisplacementDecoderSourcePolicy,
)
from core.decoding.decoder_cache import (
    build_decoder_cache_identity,
    build_decoder_cache_path,
    load_decoder_cache,
    load_decoder_cache_source_identity,
    resolve_current_residual_source_identity,
    resolve_local_residual_source_identity,
    resolve_public_residual_source_identity,
    save_decoder_cache_source_identity,
    save_decoder_provenance,
    save_decoder_cache,
)
from core.residual_field.commit import RESIDUAL_FIELD_STAGE
from core.storage.attempt_store import stage_commit_path
from core.decoding.commit import DecoderCommitManifest, decoder_commit_path, write_decoder_commit
from core.decoding.payloads import build_decoding_payload
from core.decoding.displacement_inputs import (
    PATCH_SPEC_FEATURE_VERSION,
    DisplacementDecoderKey,
    DisplacementPatchSpec,
    _decoder_assignment_mode,
    _decoder_family,
    _decoder_feature_dims,
    _has_any_decoder,
    _has_decoder_family,
    _has_single_decoder,
    _normalize_patch_axis,
    _reference_number_for_point,
    _site_class_key_for_point,
    _stack_features_into_columns,
    apply_decoder,
    apply_decoder_family,
    build_displacement_decoder_key,
    build_displacement_patch_spec,
    build_feature_sets,
    collect_displacement_decoder_keys,
    ensure_decoder,
    prepare_displacement_decoder_inputs,
    validate_global_displacement_patch_specs,
)
from core.decoding.loader import resolve_output_dir
from core.decoding.decode_chunk import PointDataPostprocessingProcessor
from core.decoding.state import build_postprocessing_processor_state
from core.models import WorkflowParameters
from core.patch_centers.contracts import PointSelectionRequest
from core.storage.digests import digest_dict
from core.storage.fingerprint import file_sha256
from core.storage.manifest import read_manifest


def _set_single_decoder(processor, decoder_M, feature_dim: int) -> None:
    processor._decoder_M = np.asarray(decoder_M, dtype=np.float64)
    processor._feature_dim = int(feature_dim)
    processor._decoder_family = None
    processor._decoder_feature_dims = None


def _set_decoder_family(
    processor,
    decoder_family: dict[DisplacementDecoderKey, np.ndarray],
    feature_dims: dict[DisplacementDecoderKey, int],
) -> None:
    processor._decoder_family = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in decoder_family.items()
    }
    processor._decoder_feature_dims = {
        key: int(feature_dims[key])
        for key in decoder_family
    }
    processor._decoder_M = None
    processor._feature_dim = None


def _unique_decoder_keys(parameters: dict, *, point_data_list) -> list[DisplacementDecoderKey]:
    keys = collect_displacement_decoder_keys(parameters, point_data_list=point_data_list)
    unique_keys = []
    seen = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    return unique_keys


def build_decoder_training_payload(
    processor,
    *,
    chunk_id,
    rifft_saver,
    point_data_list,
    output_dir=None,
):
    output_dir = resolve_output_dir(rifft_saver, chunk_id, output_dir)
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
            "use processing.decoder.source='compute' or 'current'."
        )
    return decoder, feature_dim


def _compute_decoder_cache_matches_residual_source(
    cache_path: str,
    *,
    source_dir,
    logger,
) -> bool:
    """Decide whether a compute-mode decoder cache may be reused.

    The cache filename encodes only the configuration hash, so it can outlive
    the residual artifacts it was trained on (a residual recompute in the same
    compute directory keeps the same name). Reuse is allowed only when the
    recorded residual-source identity matches the artifacts currently on disk.
    When no residual artifacts exist to compare against (the compute directory
    was cleaned down to the cache), the identity is unverifiable and the cache
    is trusted as before.
    """
    current_identity = resolve_local_residual_source_identity(output_dir=source_dir)
    if current_identity is None:
        # The compute directory was cleaned down to the cache: nothing left to
        # verify against, and the cache filename encodes only geometry and
        # hyperparameters -- NOT the input structure data. If the input data
        # changed since training, this decoder is silently wrong.
        logger.warning(
            "Compute-mode decoder cache at '%s' cannot be verified: no residual "
            "artifacts remain in '%s'. Reusing it. If the input structure or "
            "data files changed since this decoder was trained, set "
            "processing.decoder.fresh_start=true (or delete the cache) to force "
            "retraining.",
            cache_path,
            source_dir,
        )
        return True
    recorded_identity = load_decoder_cache_source_identity(cache_path)
    if recorded_identity is None:
        logger.warning(
            "Compute-mode decoder cache at '%s' has no recorded residual-source "
            "identity; retraining from the residual artifacts in '%s' to rule "
            "out a stale decoder.",
            cache_path,
            source_dir,
        )
        return False
    if recorded_identity.get("source_identity_digest") != current_identity.get(
        "source_identity_digest"
    ):
        logger.warning(
            "Compute-mode decoder cache at '%s' was trained from residual "
            "artifacts that no longer match '%s'; retraining.",
            cache_path,
            source_dir,
        )
        return False
    return True


def _resolve_decoder_cache_path(cache_path: str, parameters: dict) -> str:
    path = Path(cache_path)
    if not path.is_dir():
        return str(cache_path)
    candidate = Path(build_decoder_cache_path(parameters, str(path)))
    if candidate.is_file():
        return str(candidate)
    # The plain parameter-hashed name misses decoders that were saved under a
    # source-identity-hashed name (decoder.source='current' runs). The producing
    # run records the exact artifact in decoder_source_provenance.json, so a
    # consumer run (decoder.source='cache' pointing at that directory) resolves
    # through the provenance instead of failing after the full pipeline ran.
    provenance_path = path / "decoder_source_provenance.json"
    if provenance_path.is_file():
        try:
            import json

            recorded = json.loads(provenance_path.read_text(encoding="utf-8")).get(
                "decoder_cache_path"
            )
        except (OSError, ValueError):
            recorded = None
        if recorded:
            recorded_path = Path(recorded)
            if recorded_path.is_file():
                return str(recorded_path)
            dir_relative = path / recorded_path.name
            if dir_relative.is_file():
                return str(dir_relative)
    return str(candidate)


def _current_residual_run_digest(parameters: dict) -> str | None:
    runtime_info = parameters.get("runtime_info", {}) or {}
    for mapping in (parameters, runtime_info):
        if not isinstance(mapping, dict):
            continue
        value = (
            mapping.get("residual_run_digest")
            or mapping.get("residual_field_run_digest")
            or mapping.get("scattering_run_digest")
        )
        if value:
            return str(value)
    return None


def _current_public_manifest_path(
    parameters: dict,
    policy: DisplacementDecoderSourcePolicy,
) -> str | None:
    if policy.public_manifest_path:
        return str(policy.public_manifest_path)
    runtime_info = parameters.get("runtime_info", {}) or {}
    decoder_info = parameters.get("decoder", {}) or {}
    for mapping in (decoder_info, parameters, runtime_info):
        if not isinstance(mapping, dict):
            continue
        value = (
            mapping.get("public_manifest_path")
            or mapping.get("public_manifest")
            or mapping.get("manifest_path")
        )
        if value:
            return str(value)
    return None


def _digest_parameter_array(parameters: dict, key: str, *, domain: str) -> str:
    value = parameters.get(key)
    if value is None:
        return digest_dict({"present": False, "key": key}, domain=domain)
    return digest_dict(
        {
            "present": True,
            "key": key,
            "shape": list(np.asarray(value).shape),
            "values": np.asarray(value).tolist(),
        },
        domain=domain,
    )


def _decoder_target_parameters(parameters: dict) -> dict[str, object]:
    return {
        "q_window_kind": parameters.get("q_window_kind", "cheb"),
        "q_window_at_db": float(parameters.get("q_window_at_db", 100.0)),
        "edge_guard_frac": float(parameters.get("edge_guard_frac", 0.10)),
        "ls_weight_gamma": float(parameters.get("ls_weight_gamma", 0.35)),
        "dog_lambda_reg": float(parameters.get("dog_lambda_reg", 1e-3)),
        "linear_max_training_samples": parameters.get("linear_max_training_samples"),
        "decoder": dict(parameters.get("decoder", {}) or {}),
    }


def _decoder_architecture_digest(parameters: dict, *, assignment: str) -> str:
    return digest_dict(
        {
            "decoder_type": "linear_displacement",
            "assignment": str(assignment),
            "patch_feature_version": PATCH_SPEC_FEATURE_VERSION,
            "target_parameters": _decoder_target_parameters(parameters),
        },
        domain="mosaic.decoder.architecture.v1",
    )


def _build_current_decoder_cache_identity(
    *,
    parameters: dict,
    source_identity: dict,
    assignment: str,
) -> dict:
    return build_decoder_cache_identity(
        residual_source_identity=source_identity,
        coordinate_digest=_digest_parameter_array(
            parameters,
            "original_coords",
            domain="mosaic.decoder.original_coords.v1",
        ),
        average_coordinate_digest=_digest_parameter_array(
            parameters,
            "average_coords",
            domain="mosaic.decoder.average_coords.v1",
        ),
        vector_digest=_digest_parameter_array(
            parameters,
            "vectors",
            domain="mosaic.decoder.vectors.v1",
        ),
        refnumber_digest=(
            None
            if parameters.get("refnumbers") is None
            else _digest_parameter_array(
                parameters,
                "refnumbers",
                domain="mosaic.decoder.refnumbers.v1",
            )
        ),
        feature_mode=str(parameters.get("postprocessing_mode", "displacement")),
        target_parameters=_decoder_target_parameters(parameters),
        decoder_architecture_digest=_decoder_architecture_digest(
            parameters,
            assignment=assignment,
        ),
        code_version=str(parameters.get("code_version", "current")),
        schema_version=1,
    )


def _require_matching_decoder_commit(
    *,
    output_dir: str,
    run_digest: str,
    cache_path: str,
    cache_identity: dict,
) -> None:
    manifest_path = decoder_commit_path(output_dir, run_digest)
    if not manifest_path.exists():
        raise RuntimeError(
            "Current decoder cache exists without decoder_commit.json; refusing stale cache reuse."
        )
    commit = read_manifest(
        manifest_path,
        codec=DecoderCommitManifest,
        output_dir=output_dir,
    )
    relative_cache_path = Path(cache_path).resolve().relative_to(Path(output_dir).resolve()).as_posix()
    if commit.decoder_cache_path != relative_cache_path:
        raise RuntimeError("decoder_commit.json points at a different decoder cache.")
    if commit.decoder_cache_identity != cache_identity:
        raise RuntimeError("decoder_commit.json identity does not match current decoder source.")
    if commit.decoder_cache_file_sha256 != file_sha256(cache_path):
        raise RuntimeError("decoder_commit.json file hash does not match current decoder cache.")
    if int(commit.decoder_cache_nbytes) != int(Path(cache_path).stat().st_size):
        raise RuntimeError("decoder_commit.json byte size does not match current decoder cache.")


def _set_prepared_decoder_from_source(processor, *, decoder, feature_dim) -> None:
    _set_single_decoder(processor, decoder, int(feature_dim))


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
    R_data, U_data, P = _stack_decoder_training_samples(
        training_features,
        training_targets,
    )
    logger.info(
        "Training linear decoder M for %s with %d samples (P=%d).",
        label,
        U_data.shape[1],
        P,
    )
    M = _solve_linear_decoder(
        R_data=R_data,
        U_data=U_data,
        lam_reg=lam_reg,
        logger=logger,
    )
    _set_single_decoder(processor, M, P)
    logger.info("Decoder M trained (shape %s).", processor._decoder_M.shape)
    save_decoder_cache(cache_path, processor._decoder_M, processor._feature_dim, logger)


def _stack_decoder_training_samples(
    training_features: list[np.ndarray],
    training_targets: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    R_data = _stack_features_into_columns(training_features)
    U_data = _stack_features_into_columns(training_targets)
    P, N = R_data.shape
    return R_data, U_data, P


def _solve_linear_decoder(
    *,
    R_data: np.ndarray,
    U_data: np.ndarray,
    lam_reg: float,
    logger,
) -> np.ndarray:
    # The launch environment pins BLAS to one thread (correct for the dask
    # workers, which parallelize across tasks) — but this solve runs in the
    # DRIVER, where one BLAS thread means a ~4e13-FLOP kernel build + solve
    # crawls on one of 96 cores while every GPU idles. Open the pool up for
    # exactly this call.
    from threadpoolctl import threadpool_limits

    from core.runtime.cpu_resources import available_cpu_count

    with threadpool_limits(limits=available_cpu_count()):
        return _solve_linear_decoder_inner(
            R_data=R_data, U_data=U_data, lam_reg=lam_reg, logger=logger
        )


def _solve_linear_decoder_inner(
    *,
    R_data: np.ndarray,
    U_data: np.ndarray,
    lam_reg: float,
    logger,
) -> np.ndarray:
    P, N = R_data.shape
    lam = float(lam_reg)
    # Pick the smaller normal system: primal (P×P) when P ≤ N, otherwise the
    # dual / kernel form (N×N) via Rᵀ(RRᵀ+λI_P)⁻¹ = (RᵀR+λI_N)⁻¹Rᵀ. The dual
    # branch avoids allocating a (P,P) matrix when the feature dim is much
    # larger than the training-sample count.
    if P <= N:
        logger.info(
            "Solving ridge decoder in primal form: P=%d, N=%d (H is %d×%d).",
            P, N, P, P,
        )
        H = R_data @ R_data.T + lam * np.eye(P)
        UR = U_data @ R_data.T
        try:
            return np.linalg.solve(H, UR.T).T
        except np.linalg.LinAlgError:
            logger.warning("Decoder normal matrix H is singular; using pseudo-inverse.")
            return UR @ np.linalg.pinv(H, rcond=1e-12)

    logger.info(
        "Solving ridge decoder in dual form: P=%d, N=%d (kernel is %d×%d).",
        P, N, N, N,
    )
    K = R_data.T @ R_data + lam * np.eye(N)
    try:
        A = np.linalg.solve(K, R_data.T)
    except np.linalg.LinAlgError:
        logger.warning("Decoder dual matrix K is singular; using pseudo-inverse.")
        A = np.linalg.pinv(K, rcond=1e-12) @ R_data.T
    return U_data @ A


def train_decoder_family_from_samples(
    processor,
    *,
    training_features: list[np.ndarray],
    training_targets: list[np.ndarray],
    training_decoder_keys: list[DisplacementDecoderKey],
    lam_reg: float,
    logger,
    label: str,
) -> None:
    if not training_features:
        raise RuntimeError(
            "No displacement decoder training samples were collected for "
            f"{label}. A full/unmasked decoder source run is required."
        )

    grouped_indices: dict[DisplacementDecoderKey, list[int]] = {}
    for index, key in enumerate(training_decoder_keys):
        grouped_indices.setdefault(key, []).append(index)

    if len(grouped_indices) == 1:
        key = next(iter(grouped_indices))
        R_data, U_data, P = _stack_decoder_training_samples(
            training_features,
            training_targets,
        )
        decoder_family = {
            key: _solve_linear_decoder(
                R_data=R_data,
                U_data=U_data,
                lam_reg=lam_reg,
                logger=logger,
            )
        }
        feature_dims = {key: P}
        _set_decoder_family(processor, decoder_family, feature_dims)
        logger.info(
            "Decoder family trained with 1 key; family mode remains active for %s.",
            label,
        )
        return

    decoder_family: dict[DisplacementDecoderKey, np.ndarray] = {}
    feature_dims: dict[DisplacementDecoderKey, int] = {}
    for key in sorted(grouped_indices, key=repr):
        indices = grouped_indices[key]
        key_features = [training_features[index] for index in indices]
        key_targets = [training_targets[index] for index in indices]
        R_data, U_data, P = _stack_decoder_training_samples(key_features, key_targets)
        logger.info(
            "Training decoder family member for %s with key %s using %d samples (P=%d).",
            label,
            key.to_mapping(),
            U_data.shape[1],
            P,
        )
        decoder_family[key] = _solve_linear_decoder(
            R_data=R_data,
            U_data=U_data,
            lam_reg=lam_reg,
            logger=logger,
        )
        feature_dims[key] = P
    _set_decoder_family(processor, decoder_family, feature_dims)
    logger.info("Decoder family trained with %d keys.", len(decoder_family))


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
        # A reused processor must not carry a stale prepared-inputs cache
        # into a different source mode/run.
        processor.prepared_inputs_cache = None
        unique_decoder_keys: list[DisplacementDecoderKey] = []
        db_manager = getattr(artifacts, "db_manager", None)
        if db_manager is not None and hasattr(db_manager, "get_pending_chunk_ids"):
            pending_chunk_ids = sorted(db_manager.get_pending_chunk_ids())
            point_data_list = []
            for chunk_id in pending_chunk_ids:
                point_data_list.extend(db_manager.get_point_data_for_chunk(int(chunk_id)))
            unique_decoder_keys = _unique_decoder_keys(
                processor.parameters,
                point_data_list=point_data_list,
            )
            if policy.assignment == "single":
                validate_global_displacement_patch_specs(
                    processor.parameters,
                    point_data_list=point_data_list,
                )

        if policy.mode == "error":
            if (
                policy.assignment == "family"
                and len(unique_decoder_keys) > 1
                and not _has_decoder_family(processor)
            ):
                raise RuntimeError(
                    "Decoder-family assignment mode requires a prepared decoder family "
                    "when multiple decoder keys are present. Preload a decoder family "
                    "in-memory or use processing.decoder.source='compute'."
                )
            if _has_single_decoder(processor) or _has_decoder_family(processor):
                provenance = DecoderSourceProvenance(
                    mode="error",
                    semantics="preloaded-family" if _has_decoder_family(processor) else "preloaded",
                    decoder_cache_path="<in-memory>",
                    feature_dim=processor._feature_dim if _has_single_decoder(processor) else None,
                    loaded_from_cache=False,
                    computed=False,
                )
                processor.decoder_source_provenance = provenance
                save_decoder_provenance(output_dir, provenance.to_mapping(), logger)
                return provenance
            raise RuntimeError(
                "Displacement decoding now requires an explicit decoder source. "
                "Set processing.decoder.source to 'cache' with a valid cache_path, "
                "to 'compute' with a separate compute_output_directory, or to "
                "'current' to train from current residual artifacts. "
                "No implicit M-decoder training is performed; this is especially "
                "important for expensive 3D runs."
            )

        if policy.mode == "current":
            # Training and decode share this processor/output: carry the
            # training pass's prepared per-chunk inputs to the decode pass.
            from core.decoding.prepared_inputs_cache import (
                PreparedInputsCache,
                prepared_cache_max_bytes,
            )

            processor.prepared_inputs_cache = PreparedInputsCache(
                max_bytes=prepared_cache_max_bytes(),
                spill_dir=output_dir,
            )
            return self._prepare_current_decoder_cache(
                processor=processor,
                policy=policy,
                artifacts=artifacts,
                output_dir=output_dir,
                logger=logger,
            )

        if policy.mode == "cache":
            if policy.assignment == "family" and len(unique_decoder_keys) > 1:
                raise RuntimeError(
                    "Decoder-family cache loading is not supported until Stage 3. "
                    "Use processing.decoder.assignment='family' together with "
                    "processing.decoder.source='compute', or preload a decoder "
                    "family in-memory."
                )
            resolved_cache_path = _resolve_decoder_cache_path(
                str(policy.cache_path),
                processor.parameters,
            )
            decoder, feature_dim = load_required_decoder(resolved_cache_path, logger)
            _set_prepared_decoder_from_source(
                processor,
                decoder=decoder,
                feature_dim=feature_dim,
            )
            provenance = DecoderSourceProvenance(
                mode="cache",
                semantics="precomputed",
                decoder_cache_path=str(resolved_cache_path),
                source_output_directory=str(Path(resolved_cache_path).resolve().parent),
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
        if cache_path is not None:
            decoder, feature_dim = load_required_decoder(cache_path, logger)
            _set_prepared_decoder_from_source(
                processor,
                decoder=decoder,
                feature_dim=feature_dim,
            )
            published_cache_path = build_decoder_cache_path(
                processor.parameters,
                output_dir,
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            if Path(published_cache_path).resolve() != Path(cache_path).resolve():
                save_decoder_cache(
                    published_cache_path,
                    processor._decoder_M,
                    processor._feature_dim,
                    logger,
                )
            cache_path = published_cache_path
        finalized = DecoderSourceProvenance(
            mode="compute",
            semantics="unmasked-family" if _has_decoder_family(processor) else "unmasked",
            decoder_cache_path=str(cache_path) if cache_path is not None else "<stage2-in-memory-family>",
            source_output_directory=(
                str(Path(cache_path).resolve().parent)
                if cache_path is not None
                else provenance.source_output_directory
            ),
            compute_output_directory=provenance.compute_output_directory,
            feature_dim=processor._feature_dim if _has_single_decoder(processor) else None,
            loaded_from_cache=provenance.loaded_from_cache,
            computed=provenance.computed,
        )
        processor.decoder_source_provenance = finalized
        save_decoder_provenance(output_dir, finalized.to_mapping(), logger)
        return finalized

    def _collect_decoder_training_samples(
        self,
        *,
        training_processor,
        artifacts,
        output_dir: str,
        prepared_cache=None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[DisplacementDecoderKey]]:
        training_features: list[np.ndarray] = []
        training_targets: list[np.ndarray] = []
        training_decoder_keys: list[DisplacementDecoderKey] = []
        max_train = training_processor.parameters.get("linear_max_training_samples")
        for chunk_id in sorted(artifacts.db_manager.get_pending_chunk_ids()):
            point_data_list = artifacts.db_manager.get_point_data_for_chunk(int(chunk_id))
            training_payload = build_decoder_training_payload(
                training_processor,
                chunk_id=int(chunk_id),
                rifft_saver=artifacts.saver,
                point_data_list=point_data_list,
                output_dir=output_dir,
            )
            # Hand the full per-site features to the decode pass (which
            # previously recomputed them from scratch: another 6.78 GB
            # residual read + grid rebuild + feature extraction per chunk).
            # .get() throughout: tests stub this payload with the training
            # subset only.
            features_all = training_payload.get("features_all")
            if prepared_cache is not None and features_all is not None:
                from core.decoding.prepared_inputs_cache import (
                    PreparedChunkInputs,
                    prepared_inputs_token,
                )

                prepared_cache.put(
                    int(chunk_id),
                    PreparedChunkInputs(
                        output_dir=str(
                            training_payload.get("output_dir", output_dir)
                        ),
                        token=prepared_inputs_token(
                            int(chunk_id), point_data_list, output_dir
                        ),
                        cids_all=training_payload.get("cids_all", []),
                        decoder_keys_all=training_payload.get(
                            "decoder_keys_all", []
                        ),
                        features_all=features_all,
                        nbytes=sum(
                            int(feature.nbytes) for feature in features_all
                        ),
                    ),
                )
            # The 6.78 GB amplitudes dict has zero readers — release it now
            # rather than at the next loop iteration's rebind.
            training_payload.pop("data", None)
            training_features.extend(training_payload["features_train"])
            training_targets.extend(training_payload["u_train"])
            training_decoder_keys.extend(training_payload["training_decoder_keys"])
        if max_train is not None:
            limit = int(max_train)
            training_features = training_features[:limit]
            training_targets = training_targets[:limit]
            training_decoder_keys = training_decoder_keys[:limit]
        return training_features, training_targets, training_decoder_keys

    def _train_decoder_from_existing_artifacts(
        self,
        *,
        target_processor,
        training_processor,
        policy: DisplacementDecoderSourcePolicy,
        artifacts,
        output_dir: str,
        cache_path: str,
        mode: str,
        single_semantics: str,
        family_semantics: str,
        compute_output_directory: str | None,
        logger,
        label: str,
    ) -> tuple[str | None, DecoderSourceProvenance]:
        (
            training_features,
            training_targets,
            training_decoder_keys,
        ) = self._collect_decoder_training_samples(
            training_processor=training_processor,
            artifacts=artifacts,
            output_dir=output_dir,
            # Reuse is only safe when training and decode share the SAME
            # processor/output (mode='current'); 'compute' trains on a
            # different, unmasked point set and must never populate it.
            prepared_cache=(
                getattr(target_processor, "prepared_inputs_cache", None)
                if training_processor is target_processor
                else None
            ),
        )
        if policy.assignment == "family":
            train_decoder_family_from_samples(
                training_processor,
                training_features=training_features,
                training_targets=training_targets,
                training_decoder_keys=training_decoder_keys,
                lam_reg=float(training_processor.parameters.get("dog_lambda_reg", 1e-3)),
                logger=logger,
                label=label,
            )
            target_processor._decoder_family = getattr(training_processor, "_decoder_family", None)
            target_processor._decoder_feature_dims = getattr(training_processor, "_decoder_feature_dims", None)
            target_processor._decoder_M = getattr(training_processor, "_decoder_M", None)
            target_processor._feature_dim = getattr(training_processor, "_feature_dim", None)
            provenance = DecoderSourceProvenance(
                mode=mode,
                semantics=family_semantics,
                decoder_cache_path="<stage2-in-memory-family>",
                source_output_directory=str(Path(output_dir).resolve()),
                compute_output_directory=compute_output_directory,
                feature_dim=None,
                loaded_from_cache=False,
                computed=True,
            )
            save_decoder_provenance(output_dir, provenance.to_mapping(), logger)
            return None, provenance

        train_decoder_from_samples(
            training_processor,
            cache_path=cache_path,
            training_features=training_features,
            training_targets=training_targets,
            lam_reg=float(training_processor.parameters.get("dog_lambda_reg", 1e-3)),
            logger=logger,
            label=label,
        )
        _set_prepared_decoder_from_source(
            target_processor,
            decoder=training_processor._decoder_M,
            feature_dim=training_processor._feature_dim,
        )
        provenance = DecoderSourceProvenance(
            mode=mode,
            semantics=single_semantics,
            decoder_cache_path=str(cache_path),
            source_output_directory=str(Path(output_dir).resolve()),
            compute_output_directory=compute_output_directory,
            feature_dim=target_processor._feature_dim,
            loaded_from_cache=False,
            computed=True,
        )
        save_decoder_provenance(output_dir, provenance.to_mapping(), logger)
        return cache_path, provenance

    def _prepare_current_decoder_cache(
        self,
        *,
        processor,
        policy: DisplacementDecoderSourcePolicy,
        artifacts,
        output_dir: str,
        logger,
    ) -> DecoderSourceProvenance:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        public_manifest_path = _current_public_manifest_path(processor.parameters, policy)
        if public_manifest_path:
            source_identity = resolve_public_residual_source_identity(
                output_dir=output_dir,
                public_manifest_path=public_manifest_path,
            )
            run_digest = str(source_identity["run_digest"])
            single_semantics = "current-public-residual"
            family_semantics = "current-public-residual-family"
        else:
            run_digest = _current_residual_run_digest(processor.parameters)
            manifest_available = run_digest is not None and stage_commit_path(
                Path(output_dir), str(run_digest), RESIDUAL_FIELD_STAGE
            ).exists()
            if manifest_available:
                source_identity = resolve_current_residual_source_identity(
                    output_dir=output_dir,
                    run_digest=run_digest,
                )
            else:
                # Local (loose-file) residual layout: no run-scoped stage_commit
                # manifests exist, so derive the source identity from the loose
                # residual_chunk_* artifacts themselves. Any residual recompute
                # changes the identity, preserving decoder-cache staleness
                # semantics.
                source_identity = resolve_local_residual_source_identity(
                    output_dir=output_dir
                )
                if source_identity is None:
                    raise RuntimeError(
                        "processing.decoder.source='current' requires either a "
                        "residual_field stage_commit manifest (with "
                        "residual_run_digest) or local residual_chunk_* "
                        "artifacts in the output directory."
                    )
                run_digest = str(source_identity["run_digest"])
            single_semantics = "current-residual"
            family_semantics = "current-residual-family"
        cache_identity = _build_current_decoder_cache_identity(
            parameters=processor.parameters,
            source_identity=source_identity,
            assignment=policy.assignment,
        )
        cache_path = build_decoder_cache_path(
            processor.parameters,
            output_dir,
            source_identity=cache_identity,
        )
        force_decoder_fresh = bool(policy.fresh_start)
        if (
            policy.assignment == "single"
            and not force_decoder_fresh
            and Path(cache_path).is_file()
        ):
            _require_matching_decoder_commit(
                output_dir=output_dir,
                run_digest=run_digest,
                cache_path=cache_path,
                cache_identity=cache_identity,
            )
            decoder, feature_dim = load_required_decoder(cache_path, logger)
            _set_prepared_decoder_from_source(
                processor,
                decoder=decoder,
                feature_dim=feature_dim,
            )
            provenance = DecoderSourceProvenance(
                mode="current",
                semantics=single_semantics,
                decoder_cache_path=str(cache_path),
                source_output_directory=str(Path(output_dir).resolve()),
                feature_dim=feature_dim,
                loaded_from_cache=True,
                computed=False,
            )
            processor.decoder_source_provenance = provenance
            save_decoder_provenance(output_dir, provenance.to_mapping(), logger)
            return provenance

        if not hasattr(artifacts, "db_manager") or not hasattr(artifacts, "saver"):
            raise RuntimeError(
                "processing.decoder.source='current' requires current-run residual "
                "artifacts and a residual-field saver. Run the ALL case through "
                "residual-field generation before rebuilding the decoder."
            )

        cache_path, provenance = self._train_decoder_from_existing_artifacts(
            target_processor=processor,
            training_processor=processor,
            policy=policy,
            artifacts=artifacts,
            output_dir=output_dir,
            cache_path=cache_path,
            mode="current",
            single_semantics=single_semantics,
            family_semantics=family_semantics,
            compute_output_directory=None,
            logger=logger,
            label="current residual decoder-source run",
        )
        if cache_path is not None and policy.assignment == "single":
            write_decoder_commit(
                output_dir=output_dir,
                run_digest=run_digest,
                decoder_cache_path=cache_path,
                decoder_cache_identity=cache_identity,
            )
        processor.decoder_source_provenance = provenance
        return provenance

    def _compute_decoder_cache(
        self,
        *,
        processor,
        policy: DisplacementDecoderSourcePolicy,
        workflow_parameters,
        structure,
        client,
    ) -> tuple[str | None, DecoderSourceProvenance]:
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
            decoder_fresh_start=policy.fresh_start,
        )
        compute_processed_dir = compute_root / "processed_point_data"
        cache_path = build_decoder_cache_path(
            processor.parameters,
            str(compute_processed_dir),
        )
        force_decoder_fresh = bool(policy.fresh_start)
        if (
            policy.assignment == "single"
            and not force_decoder_fresh
            and Path(cache_path).is_file()
            and _compute_decoder_cache_matches_residual_source(
                cache_path,
                source_dir=compute_processed_dir,
                logger=logger,
            )
        ):
            decoder, feature_dim = load_required_decoder(cache_path, logger)
            _set_prepared_decoder_from_source(
                processor,
                decoder=decoder,
                feature_dim=feature_dim,
            )
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
        if force_decoder_fresh and compute_processed_dir.exists():
            shutil.rmtree(compute_processed_dir)
        compute_processed_dir.mkdir(parents=True, exist_ok=True)
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
            training_processor = PointDataPostprocessingProcessor(
                compute_artifacts.db_manager,
                compute_artifacts.point_data_processor,
                decoding_parameters,
            )

            trained_cache_path, provenance = self._train_decoder_from_existing_artifacts(
                target_processor=processor,
                training_processor=training_processor,
                policy=policy,
                artifacts=compute_artifacts,
                output_dir=str(compute_processed_dir),
                cache_path=cache_path,
                mode="compute",
                single_semantics="unmasked",
                family_semantics="unmasked-family",
                compute_output_directory=str(compute_root),
                logger=logger,
                label="full/unmasked decoder-source run",
            )
            if trained_cache_path is not None:
                trained_identity = resolve_local_residual_source_identity(
                    output_dir=compute_processed_dir,
                )
                if trained_identity is not None:
                    save_decoder_cache_source_identity(
                        trained_cache_path,
                        trained_identity,
                        logger,
                    )
            return trained_cache_path, provenance
        finally:
            if compute_artifacts is not None:
                compute_artifacts.close()

    def _build_unmasked_workflow_parameters(
        self,
        workflow_parameters,
        compute_root: Path,
        *,
        decoder_fresh_start: bool | None = None,
    ):
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
        rspace_info["fresh_start"] = bool(decoder_fresh_start)
        rspace_info["run_postprocessing"] = False
        original_decoder = dict(rspace_info.get("decoder", {}))
        decoder_payload = {"source": "error"}
        if "assignment" in original_decoder:
            decoder_payload["assignment"] = original_decoder["assignment"]
        rspace_info["decoder"] = decoder_payload
        payload["rspace_info"] = rspace_info
        return WorkflowParameters.from_payload(payload)
