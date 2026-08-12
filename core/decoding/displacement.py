from __future__ import annotations

import logging
import os

import numpy as np

from core.decoding.displacement_inputs import (
    apply_decoder,
    apply_decoder_family,
    ensure_decoder,
    prepare_displacement_decoder_inputs,
)
from core.decoding.io import write_displacements_csv


def compute_and_save_displacements(
    processor,
    *,
    chunk_id,
    rifft_saver,
    point_data_list,
    output_dir=None,
):
    log = logging.getLogger(__name__)
    from core.decoding.prepared_inputs_cache import prepared_inputs_token
    from core.residual_field.loader import resolve_output_dir

    resolved_dir = resolve_output_dir(rifft_saver, chunk_id, output_dir)
    cache = getattr(processor, "prepared_inputs_cache", None)
    entry = (
        None
        if cache is None
        else cache.pop(
            int(chunk_id),
            token=prepared_inputs_token(chunk_id, point_data_list, resolved_dir),
            output_dir=resolved_dir,
        )
    )
    if entry is not None:
        # Training pass (decoder.source='current') already prepared these
        # exact inputs: skip the 6.78 GB residual reload, grid rebuild and
        # per-site feature extraction the decode pass used to repeat.
        output_dir = entry.output_dir
        features_all = entry.features_all
        cids_all = entry.cids_all
        decoder_keys_all = entry.decoder_keys_all
    else:
        prepared = prepare_displacement_decoder_inputs(
            processor,
            chunk_id=chunk_id,
            rifft_saver=rifft_saver,
            point_data_list=point_data_list,
            output_dir=resolved_dir,
        )
        output_dir = prepared["output_dir"]
        features_all = prepared["features_all"]
        cids_all = prepared["cids_all"]
        decoder_keys_all = prepared["decoder_keys_all"]
    ensure_decoder(
        processor,
        features_all=features_all,
        decoder_keys_all=decoder_keys_all,
        logger=log,
    )
    if getattr(processor.decoder_source_policy, "assignment", "single") == "family":
        U_all = apply_decoder_family(processor, features_all, decoder_keys_all)
    else:
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

    h5_path = os.path.join(output_dir, f"chunk_{chunk_id}_site_displacements.h5")
    csv_path = os.path.join(output_dir, f"chunk_{chunk_id}_site_displacements.csv")
    rifft_saver.save_data(out_table, h5_path)
    write_displacements_csv(csv_path, ids, U)

    return out_table
