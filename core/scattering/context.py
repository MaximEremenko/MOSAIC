from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.scattering.coefficients import to_numpy
from core.scattering.form_factors.contracts import ScatteringWeightSelection
from core.processing_mode import normalize_processing_mode
from core.models import ReciprocalSpaceArtifacts, StructureData, WorkflowParameters


@dataclass(frozen=True)
class ScatteringExecutionContext:
    workflow_parameters: WorkflowParameters
    structure: StructureData
    artifacts: ReciprocalSpaceArtifacts
    dimension: int
    postprocessing_mode: str
    unsaved_interval_chunks: list[tuple[int, int]]
    point_rows: list[dict[str, object]]
    intervals: list[dict[str, object]]
    chemical_filtered: bool
    use_coeff: bool
    centered_coefficients: np.ndarray
    mask_strategy: object
    scattering_weight_selection: ScatteringWeightSelection
    # Reference for the average-amplitude channel: 'factorized' (crystal
    # default), 'direct' (average_coords transformed directly — the
    # amorphous reference configuration), or 'homogeneous' (zero average;
    # everything at q != 0 is diffuse). See kernels.ReferenceSpec.
    reference_mode: str = "factorized"


def build_scattering_execution_context(
    *,
    workflow_parameters: WorkflowParameters,
    structure: StructureData,
    artifacts,
    parameter_loading_service,
    coefficient_centering_service,
    mask_strategy_service,
    interval_reconstruction_service,
) -> ScatteringExecutionContext:
    dimension = workflow_parameters.struct_info.dimension
    rspace = workflow_parameters.rspace_info
    post_mode = normalize_processing_mode(rspace.mode or "displacement")
    get_all_pairs = getattr(artifacts.db_manager, "get_interval_chunks", None)
    unsaved = (
        get_all_pairs()
        if callable(get_all_pairs)
        else artifacts.db_manager.get_unsaved_interval_chunks()
    )
    pending_work = interval_reconstruction_service.load_pending_work(
        artifacts,
        dimension,
    )
    chemical_filtered = bool(rspace.chemical_filtered_ordering)
    use_coeff = bool(rspace.use_coeff if rspace.use_coeff is not None else True) or chemical_filtered
    reference_mode = (
        str(getattr(rspace, "reference_mode", None) or "factorized")
        .strip()
        .lower()
    )
    if reference_mode not in {"factorized", "direct", "homogeneous"}:
        raise ValueError(
            f"Unknown processing.reference_mode {reference_mode!r}; expected "
            "'factorized', 'direct' or 'homogeneous'."
        )
    if reference_mode == "direct" and chemical_filtered:
        raise ValueError(
            "reference_mode 'direct' subtracts A(average_coords) from "
            "A(original_coords); chemical_filtered_ordering substitutes "
            "average_coords FOR original_coords, which would make the delta "
            "channel identically zero. The two options are incompatible."
        )
    _require_usable_mask_for_one_cell_box(
        dimension=dimension,
        supercell=to_numpy(structure.supercell),
        peak_info=workflow_parameters.peak_info,
    )
    coeff_center_mode = rspace.coeff_center_by or ("global" if chemical_filtered else "none")
    centered_coeff = coefficient_centering_service.center(
        np.asarray(to_numpy(structure.coeff), float),
        to_numpy(structure.refnumbers),
        coeff_center_mode,
    )
    return ScatteringExecutionContext(
        workflow_parameters=workflow_parameters,
        structure=structure,
        artifacts=artifacts,
        dimension=dimension,
        postprocessing_mode=post_mode,
        unsaved_interval_chunks=unsaved,
        point_rows=pending_work.point_rows,
        intervals=pending_work.intervals,
        chemical_filtered=chemical_filtered,
        use_coeff=use_coeff,
        centered_coefficients=centered_coeff,
        mask_strategy=mask_strategy_service.build(
            dimension,
            workflow_parameters.peak_info,
            post_mode=post_mode,
        ),
        scattering_weight_selection=parameter_loading_service.resolve_scattering_weight_settings(
            workflow_parameters
        ),
        reference_mode=reference_mode,
    )


def _require_usable_mask_for_one_cell_box(
    *,
    dimension: int,
    supercell,
    peak_info,
) -> None:
    """Refuse the built-in Bragg-node mask on a one-cell (amorphous) box.

    The 3D special-points fallback mask classifies q-points by distance
    from the nearest INTEGER hkl of the unit cell (``Mod(h,1.0) - 0.5``).
    With ``supercell=(1,1,1)`` every grid point IS integer hkl, the Mod
    terms are identically zero, and the mask silently rejects the entire
    grid — the run would complete with no q-points. An explicit mask
    equation (e.g. an |q| shell in box hkl) is unaffected."""
    if dimension != 3:
        return
    if not np.all(np.asarray(supercell, dtype=float) == 1):
        return
    mapping = (
        peak_info.to_mapping() if hasattr(peak_info, "to_mapping") else dict(peak_info or {})
    )
    has_equation = any(
        mapping.get(key)
        for key in ("mask_equation", "maskEquation", "equation", "condition")
    )
    special_points = mapping.get("specialPoints") or mapping.get("special_points")
    if not has_equation and isinstance(special_points, list) and special_points:
        raise ValueError(
            "The built-in special-points mask is a Bragg-node classifier "
            "(Mod(h,1.0)) and rejects every point of a one-cell box "
            "(supercell=(1,1,1), the amorphous encoding). Provide an "
            "explicit reciprocal_space.mask.equation in box hkl instead — "
            "e.g. an |q| shell: '(h**2 + k**2 + l**2 >= R1**2) & "
            "(h**2 + k**2 + l**2 <= R2**2)'."
        )


__all__ = ["ScatteringExecutionContext", "build_scattering_execution_context"]
