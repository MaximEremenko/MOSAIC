# -*- coding: utf-8 -*-

import logging

from numba import set_num_threads

from core.decoding.decoder_cache import build_decoder_cache_path
from core.decoding.displacement_service import (
    compute_and_save_displacements,
)
from core.decoding.site_intensity_service import (
    compute_and_save_site_intensities,
)
from core.decoding.state import build_postprocessing_processor_state
from core.decoding.window import filter_from_window as filter_from_window_impl

set_num_threads(32)
logger = logging.getLogger(__name__)


class PointDataPostprocessingProcessor:
    def __init__(self, db_manager, point_data_processor, parameters):
        self.db_manager = db_manager
        self.point_data_processor = point_data_processor
        state = build_postprocessing_processor_state(parameters)
        self.parameters = state.parameters
        self.mode = state.mode
        self.original_coords = state.original_coords
        self.average_coords = state.average_coords
        self.u_true_all = state.u_true_all
        self.decoder_source_policy = state.decoder_source_policy
        self.decoder_source_provenance = None
        self._decoder_M = None
        self._feature_dim = None
        self._decoder_family = None
        self._decoder_feature_dims = None

    def process_chunk(self, chunk_id, rifft_saver, output_dir):
        point_data_list = self.db_manager.get_point_data_for_chunk(chunk_id)
        if not point_data_list:
            logger.warning("No point data found for chunk_id: %s", chunk_id)
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

    def compute_and_save_displacements(
        self,
        *,
        chunk_id,
        rifft_saver,
        point_data_list,
        output_dir=None,
        broadcast_into_rows=False,
    ):
        return compute_and_save_displacements(
            self,
            chunk_id=chunk_id,
            rifft_saver=rifft_saver,
            point_data_list=point_data_list,
            output_dir=output_dir,
            broadcast_into_rows=broadcast_into_rows,
        )

    def compute_and_save_site_intensities(
        self,
        *,
        chunk_id,
        rifft_saver,
        point_data_list,
        output_dir=None,
    ):
        return compute_and_save_site_intensities(
            self,
            chunk_id=chunk_id,
            rifft_saver=rifft_saver,
            point_data_list=point_data_list,
            output_dir=output_dir,
        )

    def filter_from_window(
        self, window0, dimensionality, window1=None, window2=None, size_aver=None
    ):
        return filter_from_window_impl(
            self.parameters,
            window0,
            dimensionality,
            window1=window1,
            window2=window2,
        )
