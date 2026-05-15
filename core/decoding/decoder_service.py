from __future__ import annotations

import logging
import shutil
from pathlib import Path
from dataclasses import dataclass

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
    resolve_current_residual_source_identity,
    resolve_public_residual_source_identity,
    save_decoder_provenance,
    save_decoder_cache,
)
from core.decoding.commit import DecoderCommitManifest, decoder_commit_path, write_decoder_commit
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
from core.storage.digests import digest_dict
from core.storage.fingerprint import file_sha256
from core.storage.manifest import read_manifest


PATCH_SPEC_FEATURE_VERSION = 1


@dataclass(frozen=True)
class DisplacementPatchSpec:
    dimension: int
    dist_from_atom_center: tuple[float, ...]
    step_in_frac: tuple[float, ...]
    q_window_kind: str
    q_window_at_db: float
    edge_guard_frac: float
    ls_weight_gamma: float
    feature_version: int = PATCH_SPEC_FEATURE_VERSION

    def to_mapping(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "dist_from_atom_center": list(self.dist_from_atom_center),
            "step_in_frac": list(self.step_in_frac),
            "q_window_kind": self.q_window_kind,
            "q_window_at_db": self.q_window_at_db,
            "edge_guard_frac": self.edge_guard_frac,
            "ls_weight_gamma": self.ls_weight_gamma,
            "feature_version": self.feature_version,
        }


@dataclass(frozen=True)
class DisplacementDecoderKey:
    site_class_key: str
    patch_spec: DisplacementPatchSpec

    def to_mapping(self) -> dict[str, object]:
        return {
            "site_class_key": self.site_class_key,
            "patch_spec": self.patch_spec.to_mapping(),
        }


def _normalize_patch_axis(values, *, dim: int) -> tuple[float, ...]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != dim:
        raise ValueError(
            f"Patch specification axis length mismatch: expected {dim}, got {arr.size}."
        )
    return tuple(float(np.round(value, 12)) for value in arr.tolist())


def build_displacement_patch_spec(parameters: dict, point_data: dict) -> DisplacementPatchSpec:
    coordinates = np.asarray(point_data["coordinates"], dtype=float).reshape(-1)
    dimension = int(coordinates.size)
    return DisplacementPatchSpec(
        dimension=dimension,
        dist_from_atom_center=_normalize_patch_axis(
            point_data["dist_from_atom_center"],
            dim=dimension,
        ),
        step_in_frac=_normalize_patch_axis(
            point_data["step_in_frac"],
            dim=dimension,
        ),
        q_window_kind=str(parameters.get("q_window_kind", "cheb")).lower(),
        q_window_at_db=float(parameters.get("q_window_at_db", 100.0)),
        edge_guard_frac=float(parameters.get("edge_guard_frac", 0.10)),
        ls_weight_gamma=float(parameters.get("ls_weight_gamma", 0.35)),
    )


def _reference_number_for_point(point_data: dict, refnumbers_all) -> int | None:
    if refnumbers_all is None:
        return None
    central_point_id = point_data.get("central_point_id")
    if central_point_id is None:
        return None
    try:
        point_id = int(central_point_id)
    except (TypeError, ValueError):
        return None
    refnumbers = np.asarray(refnumbers_all).reshape(-1)
    if point_id < 0 or point_id >= refnumbers.shape[0]:
        return None
    return int(refnumbers[point_id])


def _site_class_key_for_point(point_data: dict, refnumbers_all) -> str:
    refnumber = _reference_number_for_point(point_data, refnumbers_all)
    if refnumber is not None:
        return f"reference_number:{refnumber}"
    central_point_id = point_data.get("central_point_id")
    return f"central_point_id:{int(central_point_id)}"


def build_displacement_decoder_key(
    parameters: dict,
    point_data: dict,
) -> DisplacementDecoderKey:
    refnumbers_all = parameters.get("refnumbers", None)
    return DisplacementDecoderKey(
        site_class_key=_site_class_key_for_point(point_data, refnumbers_all),
        patch_spec=build_displacement_patch_spec(parameters, point_data),
    )


def collect_displacement_decoder_keys(
    parameters: dict,
    *,
    point_data_list,
) -> list[DisplacementDecoderKey]:
    return [
        build_displacement_decoder_key(parameters, point_data)
        for point_data in point_data_list
    ]


def validate_global_displacement_patch_specs(
    parameters: dict,
    *,
    point_data_list,
) -> DisplacementPatchSpec | None:
    if not point_data_list:
        return None

    patch_specs = [build_displacement_patch_spec(parameters, point_data) for point_data in point_data_list]
    unique_specs = []
    seen_specs = set()
    for spec in patch_specs:
        if spec not in seen_specs:
            seen_specs.add(spec)
            unique_specs.append(spec)

    refnumbers_all = parameters.get("refnumbers", None)
    specs_by_refnumber: dict[int, set[DisplacementPatchSpec]] = {}
    for point_data, spec in zip(point_data_list, patch_specs):
        refnumber = _reference_number_for_point(point_data, refnumbers_all)
        if refnumber is None:
            continue
        specs_by_refnumber.setdefault(refnumber, set()).add(spec)

    conflicting_refnumbers = {
        refnumber: specs
        for refnumber, specs in specs_by_refnumber.items()
        if len(specs) > 1
    }
    if conflicting_refnumbers:
        refnumber, specs = next(iter(conflicting_refnumbers.items()))
        spec_details = [spec.to_mapping() for spec in sorted(specs, key=lambda item: repr(item))]
        raise ValueError(
            "Displacement decoder patch-spec inconsistency: referenceNumber "
            f"{refnumber} appears with multiple patch specs in single-global-decoder "
            "mode. This is invalid because the same site class cannot use multiple "
            "patch operators in one global decoder run. Use one shared patch spec "
            "for this referenceNumber now; the proper future extension is a fixed "
            "decoder family keyed by site class + patch spec, not per-mask retraining. "
            f"Conflicting patch specs: {spec_details}"
        )

    if len(unique_specs) > 1:
        raise ValueError(
            "Displacement decoder patch-spec inconsistency: single-global-decoder "
            "mode requires exactly one patch spec, but multiple patch specs were "
            "found across the decoded points. Mixed patch specs are not allowed in "
            "the current global-decoder path. The proper future extension is a "
            "fixed decoder family keyed by site class + patch spec, not per-mask "
            f"retraining. Found patch specs: {[spec.to_mapping() for spec in unique_specs]}"
        )

    return unique_specs[0]


def _decoder_assignment_mode(processor) -> str:
    policy = getattr(processor, "decoder_source_policy", None)
    assignment = getattr(policy, "assignment", None)
    if assignment in {"single", "family"}:
        return str(assignment)
    decoder_mapping = processor.parameters.get("decoder", {})
    if isinstance(decoder_mapping, dict):
        assignment = decoder_mapping.get("assignment")
        if assignment in {"single", "family"}:
            return str(assignment)
    return "single"


def _has_single_decoder(processor) -> bool:
    return (
        getattr(processor, "_decoder_M", None) is not None
        and getattr(processor, "_feature_dim", None) is not None
    )


def _decoder_family(processor) -> dict[DisplacementDecoderKey, np.ndarray] | None:
    family = getattr(processor, "_decoder_family", None)
    return family if family else None


def _decoder_feature_dims(processor) -> dict[DisplacementDecoderKey, int] | None:
    feature_dims = getattr(processor, "_decoder_feature_dims", None)
    return feature_dims if feature_dims else None


def _has_decoder_family(processor) -> bool:
    family = _decoder_family(processor)
    feature_dims = _decoder_feature_dims(processor)
    return bool(family) and bool(feature_dims)


def _has_any_decoder(processor) -> bool:
    return _has_single_decoder(processor) or _has_decoder_family(processor)


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
    decoder_keys = collect_displacement_decoder_keys(
        processor.parameters,
        point_data_list=point_data_list,
    )
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
    decoder_keys_all = []
    features_train = []
    u_train = []
    training_decoder_keys = []

    for point_data, decoder_key in zip(point_data_list, decoder_keys):
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
        decoder_keys_all.append(decoder_key)

        if not _has_any_decoder(processor):
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
            training_decoder_keys.append(decoder_key)

    return (
        features_all,
        cids_all,
        decoder_keys_all,
        features_train,
        u_train,
        training_decoder_keys,
    )


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
            "use processing.decoder.source='compute' or 'current'."
        )
    return decoder, feature_dim


def _resolve_decoder_cache_path(cache_path: str, parameters: dict) -> str:
    path = Path(cache_path)
    if path.is_dir():
        return build_decoder_cache_path(parameters, str(path))
    return str(path)


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


def _stack_features_into_columns(features: list[np.ndarray]) -> np.ndarray:
    """Preallocate an (P, N) column stack. Bitwise-identical to
    ``np.stack(features, axis=1)`` for 1-D inputs: only copies bytes into a
    pre-sized buffer, no arithmetic or float-reordering."""
    P = int(features[0].size)
    N = len(features)
    out = np.empty((P, N), dtype=features[0].dtype)
    for i, f in enumerate(features):
        out[:, i] = f
    return out


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


def ensure_decoder(
    processor,
    *,
    features_all,
    decoder_keys_all=None,
    logger=None,
):
    if not features_all:
        raise RuntimeError("No site features constructed; nothing to do.")
    assignment = _decoder_assignment_mode(processor)
    if assignment == "family":
        family = _decoder_family(processor)
        feature_dims = _decoder_feature_dims(processor)
        if not family and _has_single_decoder(processor):
            unique_keys = []
            seen = set()
            for key in decoder_keys_all or ():
                if key not in seen:
                    seen.add(key)
                    unique_keys.append(key)
            if len(unique_keys) <= 1:
                if any(feature.size != processor._feature_dim for feature in features_all):
                    raise RuntimeError(
                        "Feature dimension mismatch: decoder expects "
                        f"{processor._feature_dim}, but some features differ."
                    )
                return
        if not family or not feature_dims:
            raise RuntimeError(
                "No prepared M-decoder family is available for displacement decoding. "
                "Family assignment mode requires one decoder per (site class, patch spec) key."
            )
        if decoder_keys_all is None:
            raise RuntimeError("Decoder keys are required for decoder-family validation.")
        missing = [
            key.to_mapping()
            for key in decoder_keys_all
            if key not in family or key not in feature_dims
        ]
        if missing:
            raise RuntimeError(
                "No prepared decoder family member is available for some displacement "
                f"decoder keys: {missing}"
            )
        for feature, key in zip(features_all, decoder_keys_all):
            expected_dim = int(feature_dims[key])
            if feature.size != expected_dim:
                raise RuntimeError(
                    "Feature dimension mismatch for decoder family key "
                    f"{key.to_mapping()}: decoder expects {expected_dim}, "
                    f"but got {feature.size}."
                )
        return

    if processor._decoder_M is None or processor._feature_dim is None:
        raise RuntimeError(
            "No prepared M-decoder is available for displacement decoding. "
            "Set processing.decoder.source to 'cache' with a valid cache_path, "
            "or use processing.decoder.source='compute' with a separate "
            "compute_output_directory, or use processing.decoder.source='current' "
            "to train from current residual artifacts."
        )

    if processor._feature_dim is None:
        processor._feature_dim = processor._decoder_M.shape[1]
    if any(f.size != processor._feature_dim for f in features_all):
        raise RuntimeError(
            "Feature dimension mismatch: decoder expects "
            f"{processor._feature_dim}, but some features differ."
        )


def apply_decoder(processor, features_all):
    assignment = _decoder_assignment_mode(processor)
    if assignment == "family" and not _has_single_decoder(processor):
        raise RuntimeError(
            "Decoder-family application requires decoder keys. "
            "Call apply_decoder(...) with decoder_keys_all in family mode."
        )
    R_all = _stack_features_into_columns(features_all)
    return (processor._decoder_M @ R_all).T


def apply_decoder_family(processor, features_all, decoder_keys_all):
    family = _decoder_family(processor)
    if not family:
        if _has_single_decoder(processor):
            return apply_decoder(processor, features_all)
        raise RuntimeError("No decoder family is prepared.")
    output_dim = next(iter(family.values())).shape[0]
    U_all = np.zeros((len(features_all), output_dim), dtype=np.float64)
    grouped_indices: dict[DisplacementDecoderKey, list[int]] = {}
    for index, key in enumerate(decoder_keys_all):
        grouped_indices.setdefault(key, []).append(index)
    for key, indices in grouped_indices.items():
        decoder_M = family[key]
        R_group = _stack_features_into_columns([features_all[index] for index in indices])
        U_group = (decoder_M @ R_group).T
        for position, index in enumerate(indices):
            U_all[index, :] = U_group[position, :]
    return U_all


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
            if run_digest is None:
                raise RuntimeError(
                    "processing.decoder.source='current' requires residual_run_digest "
                    "or residual_field_run_digest before cache lookup."
                )
            source_identity = resolve_current_residual_source_identity(
                output_dir=output_dir,
                run_digest=run_digest,
            )
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
            from core.decoding.processor import PointDataPostprocessingProcessor

            training_processor = PointDataPostprocessingProcessor(
                compute_artifacts.db_manager,
                compute_artifacts.point_data_processor,
                decoding_parameters,
            )

            return self._train_decoder_from_existing_artifacts(
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
        from core.models import WorkflowParameters

        return WorkflowParameters.from_payload(payload)
