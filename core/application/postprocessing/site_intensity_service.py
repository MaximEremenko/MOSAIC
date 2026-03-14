from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from core.application.postprocessing.io import write_site_intensities_csv
from core.application.postprocessing.loader import (
    load_chunk_amplitudes_and_grid,
    resolve_output_dir,
)


def compute_and_save_site_intensities(
    processor,
    *,
    chunk_id,
    rifft_saver,
    point_data_list,
    output_dir=None,
):
    output_dir = resolve_output_dir(rifft_saver, chunk_id, output_dir)
    _, amplitudes, rifft_space_grid = load_chunk_amplitudes_and_grid(
        processor,
        chunk_id=chunk_id,
        point_data_list=point_data_list,
        rifft_saver=rifft_saver,
        logger=None,
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

    elements_all = processor.parameters.get("elements", None)
    refnumbers_all = processor.parameters.get("refnumbers", None)
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

    h5_path = os.path.join(output_dir, f"output_chunk_{chunk_id}_chemical_site_intensities.h5")
    csv_path = os.path.join(output_dir, f"output_chunk_{chunk_id}_chemical_site_intensities.csv")
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
