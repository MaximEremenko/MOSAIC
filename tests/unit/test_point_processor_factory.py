import pytest

from core.factories.point_processor_factory import PointProcessorFactory
from core.processors.full_list_point_processor import FullListPointProcessor


def test_point_processor_factory_creates_full_list_processor():
    processor = PointProcessorFactory.create_processor(
        "full_list",
        {"rspace_info": {"num_chunks": 2}},
        average_structure={"vectors": None},
    )
    assert isinstance(processor, FullListPointProcessor)


def test_point_processor_factory_rejects_unknown_method():
    with pytest.raises(ValueError):
        PointProcessorFactory.create_processor(
            "unknown",
            {"rspace_info": {}},
            average_structure={"vectors": None},
        )
