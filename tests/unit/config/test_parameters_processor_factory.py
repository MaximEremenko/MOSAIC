from core.config.factories.parameter_factory import (
    ParametersProcessorFactory,
)
from core.storage.hdf5_parameter_storage import HDF5ParameterSaver


def test_hdf5_parameter_source_is_supported(tmp_path):
    payload = {
        "structInfo": {
            "dimension": 1,
            "working_directory": "sample",
            "filename": "sample.f1d",
        },
        "peakInfo": {},
        "rspace_info": {"method": "from_average", "num_chunks": 1, "points": []},
    }
    hdf5_path = tmp_path / "parameters.hdf5"
    HDF5ParameterSaver(str(hdf5_path)).save_data(payload)

    processor = ParametersProcessorFactory().create_processor(
        str(hdf5_path),
        source_type="hdf5",
        hdf5_file_path=str(hdf5_path),
    )
    processor.process()
    assert processor.get_parameters() == payload
