from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import numpy as np

from core.decoding.features import build_feature_vector_from_patch
from core.decoding.grid import (
    apply_rq_pipeline_local,
    center_patch_subvoxel,
    compute_hkl_max_from_intervals,
    regrid_patch_to_c,
)
from core.residual_field.loader import (
    load_chunk_residual_field_and_grid,
    resolve_output_dir,
)


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


# ---------------------------------------------------------------------------
# Decoder application helpers
# These live here (a leaf module) so that both decoder_training and
# displacement can import them without creating a circular dependency.
# ---------------------------------------------------------------------------

def _stack_features_into_columns(features: list) -> np.ndarray:
    """Preallocate an (P, N) column stack.

    Bitwise-identical to ``np.stack(features, axis=1)`` for 1-D inputs: only
    copies bytes into a pre-sized buffer, no arithmetic or float-reordering.
    """
    P = int(features[0].size)
    N = len(features)
    out = np.empty((P, N), dtype=features[0].dtype)
    for i, f in enumerate(features):
        out[:, i] = f
    return out


def apply_decoder(processor, features_all):
    """Apply the single linear decoder matrix to a list of feature vectors."""
    assignment = _decoder_assignment_mode(processor)
    if assignment == "family" and not _has_single_decoder(processor):
        raise RuntimeError(
            "Decoder-family application requires decoder keys. "
            "Call apply_decoder(...) with decoder_keys_all in family mode."
        )
    R_all = _stack_features_into_columns(features_all)
    return (processor._decoder_M @ R_all).T


def apply_decoder_family(processor, features_all, decoder_keys_all):
    """Apply a per-key decoder family to a list of feature vectors."""
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


def ensure_decoder(
    processor,
    *,
    features_all,
    decoder_keys_all=None,
    logger=None,
):
    """Validate that the processor has a usable decoder and that feature dims match."""
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


# ---------------------------------------------------------------------------
# Parallel per-site feature extraction
# ---------------------------------------------------------------------------
# The per-site pipeline (window -> regrid -> feature vector) is pure numpy on
# disjoint slices of the chunk arrays, so it parallelizes across sites with a
# fork process pool: children inherit the multi-GB chunk arrays copy-on-write
# through this module-global (never pickled), compute independently, and
# return only ~74 KB feature vectors. Results are reassembled in submission
# order, so output is byte-identical to the serial loop. Serial fallback when
# fork is unavailable or MOSAIC_DECODE_PARALLEL=1. This loop was the driver's
# single-loaded-core phase (~3 min/chunk at hkl40) on a 96-core host.
_SITE_FEATURE_CTX: dict = {}


def _site_feature_one(item) -> np.ndarray:
    cid, _decoder_key, center = item
    ctx = _SITE_FEATURE_CTX
    coords_all = ctx["coords_all"]
    Rvals_all = ctx["Rvals_all"]
    params = ctx["params"]
    idxs = np.asarray(ctx["groups"][int(cid)], int)
    coords = coords_all[idxs, :]
    Rvals = Rvals_all[idxs]
    Rvals_proc = apply_rq_pipeline_local(
        Rvals,
        coords,
        q_window_kind=params["q_window_kind"],
        q_window_at_db=params["q_window_at_db"],
        size_aver=params["size_aver"],
        hkl_max_xyz=params["hkl_max_xyz"],
        guard_frac=params["guard_frac"],
    )
    y_grid, shape, axes_vals, _ = regrid_patch_to_c(coords, Rvals_proc)
    return build_feature_vector_from_patch(
        y_grid,
        axes_vals,
        center_abs=np.asarray(center, float),
        D=len(shape),
        weight_gamma=params["weight_g"],
        remove_odd_tilt=True,
        center_patch_subvoxel=center_patch_subvoxel,
    )


def _decode_parallel_workers(n_sites: int) -> int:
    raw = os.getenv("MOSAIC_DECODE_PARALLEL")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1, min(int(raw), n_sites))
        except (TypeError, ValueError):
            pass
    from core.runtime.cpu_resources import available_cpu_count

    return max(1, min(available_cpu_count(), n_sites))


def _compute_site_features(site_items, *, coords_all, Rvals_all, groups, params):
    workers = _decode_parallel_workers(len(site_items))
    use_pool = workers > 1 and len(site_items) >= 4
    if use_pool:
        import multiprocessing

        try:
            fork_ctx = multiprocessing.get_context("fork")
        except ValueError:
            use_pool = False
    if not use_pool:
        _SITE_FEATURE_CTX.update(
            coords_all=coords_all, Rvals_all=Rvals_all, groups=groups, params=params
        )
        try:
            return [_site_feature_one(item) for item in site_items]
        finally:
            _SITE_FEATURE_CTX.clear()
    _SITE_FEATURE_CTX.update(
        coords_all=coords_all, Rvals_all=Rvals_all, groups=groups, params=params
    )
    try:
        # NOTE for CUDA-initialized parents: fork children run ONLY numpy —
        # they never touch the GPU, which keeps fork-after-CUDA safe.
        with fork_ctx.Pool(processes=workers) as pool:
            chunksize = max(1, len(site_items) // (workers * 4))
            return pool.map(_site_feature_one, site_items, chunksize=chunksize)
    finally:
        _SITE_FEATURE_CTX.clear()


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

    site_items = []
    for point_data, decoder_key in zip(point_data_list, decoder_keys):
        cid = int(point_data["central_point_id"])
        if cid not in groups:
            continue
        center = id2center.get(cid, None)
        if center is None:
            continue
        site_items.append((cid, decoder_key, center))

    site_params = {
        "q_window_kind": q_window_kind,
        "q_window_at_db": q_window_at_db,
        "size_aver": size_aver,
        "hkl_max_xyz": hkl_max_xyz,
        "guard_frac": guard_frac,
        "weight_g": weight_g,
    }
    site_features = _compute_site_features(
        site_items,
        coords_all=coords_all,
        Rvals_all=Rvals_all,
        groups=groups,
        params=site_params,
    )

    for (cid, decoder_key, _center), feat in zip(site_items, site_features):
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


def prepare_displacement_decoder_inputs(
    processor,
    *,
    chunk_id,
    rifft_saver,
    point_data_list,
    output_dir=None,
):
    output_dir = resolve_output_dir(rifft_saver, chunk_id, output_dir)
    log = logging.getLogger(__name__)
    if _decoder_assignment_mode(processor) == "single":
        validate_global_displacement_patch_specs(
            processor.parameters,
            point_data_list=point_data_list,
        )
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

    (
        features_all,
        cids_all,
        decoder_keys_all,
        features_train,
        u_train,
        training_decoder_keys,
    ) = build_feature_sets(
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

    return {
        "output_dir": output_dir,
        "data": data,
        "features_all": features_all,
        "cids_all": cids_all,
        "decoder_keys_all": decoder_keys_all,
        "features_train": features_train,
        "u_train": u_train,
        "training_decoder_keys": training_decoder_keys,
    }
