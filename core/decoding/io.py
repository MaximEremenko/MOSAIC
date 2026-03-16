from __future__ import annotations

import csv

import numpy as np


def load_scalar_from_store(rifft_saver, filename, key_candidates):
    try:
        data = rifft_saver.load_data(filename)
    except FileNotFoundError:
        return None
    for key in key_candidates:
        if key in data:
            value = np.array(data[key])
            try:
                return int(np.ravel(value)[0])
            except Exception:
                pass
    return None


def normalize_amplitudes_ntotal(amplitudes, *, rifft_saver, chunk_id, logger=None):
    filename = rifft_saver.generate_filename(
        chunk_id,
        suffix="_amplitudes_ntotal_reciprocal_space_points",
    )
    ntot = load_scalar_from_store(
        rifft_saver,
        filename,
        ["ntotal_reciprocal_space_points", "ntotal_reciprocal_points"],
    )
    if not ntot or ntot <= 0:
        if logger:
            logger.warning(
                "[normalize_amplitudes_ntotal] Missing ntotal; leaving amplitudes unscaled."
            )
        return amplitudes

    scale = 1.0 / float(ntot)
    try:
        if amplitudes.ndim == 2 and amplitudes.shape[1] >= 2:
            amplitudes[:, 1] *= scale
        else:
            amplitudes[...] *= scale
    except Exception as exc:
        if logger:
            logger.warning(
                "[normalize_amplitudes_ntotal] scaling failed: %s",
                exc,
            )
    return amplitudes


def write_displacements_csv(csv_path, ids, U):
    U = np.asarray(U, float)
    cols = ["ux", "uy", "uz"][: U.shape[1]]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["central_point_id", *cols, "units=cartesian"])
        for point_id, u in zip(ids.tolist(), U.tolist()):
            writer.writerow([point_id, *[f"{x:.6E}" for x in u], "cartesian"])


def write_site_intensities_csv(
    csv_path,
    ids,
    coords,
    intens_real,
    intens_imag,
    *,
    elements=None,
    refnumbers=None,
):
    ids = np.asarray(ids).ravel()
    coords = np.asarray(coords, float)
    intens_real = np.asarray(intens_real, float).ravel()
    intens_imag = np.asarray(intens_imag, float).ravel()

    if coords.ndim != 2:
        raise ValueError("coords must be a (N,D) array")
    dim = int(coords.shape[1])
    if dim < 1 or dim > 3:
        raise ValueError(f"Unsupported coordinate dimension D={dim}")

    coords3 = np.zeros((coords.shape[0], 3), dtype=float)
    coords3[:, :dim] = coords

    if elements is not None:
        elements = np.asarray(elements, dtype=object).ravel()
        if elements.shape[0] != ids.shape[0]:
            raise ValueError("elements length mismatch")
    if refnumbers is not None:
        refnumbers = np.asarray(refnumbers).ravel()
        if refnumbers.shape[0] != ids.shape[0]:
            raise ValueError("refnumbers length mismatch")

    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "central_point_id",
                "element",
                "refnumber",
                "x",
                "y",
                "z",
                "intensity_real",
                "intensity_imag",
            ]
        )

        for index in range(ids.shape[0]):
            element = elements[index] if elements is not None else ""
            refnumber = int(refnumbers[index]) if refnumbers is not None else ""
            x, y, z = coords3[index].tolist()
            writer.writerow(
                [
                    int(ids[index]),
                    element,
                    refnumber,
                    f"{x:.8E}",
                    f"{y:.8E}",
                    f"{z:.8E}",
                    f"{float(intens_real[index]):.8E}",
                    f"{float(intens_imag[index]):.8E}",
                ]
            )

