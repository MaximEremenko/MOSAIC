"""Streaming subchunk-slot count must not depend on live cluster state.

The slot is part of work-unit/checkpoint identity: a run started on a
1-GPU node has to resume on an 8-GPU node with its checkpoints intact,
so the default cannot be len(workers).
"""
from types import SimpleNamespace

from core.residual_field.execution import _streaming_subchunk_slot_count


class _LiveDistributedClient:
    """Looks like a live distributed client (not sync): is_sync_client
    checks for a running asyncio loop."""

    asynchronous = False

    def __init__(self):
        self.loop = SimpleNamespace(asyncio_loop=object())

    def scheduler_info(self):
        return {"workers": {f"tcp://w{i}": {} for i in range(3)}}


def _params(**runtime_info):
    return SimpleNamespace(runtime_info=runtime_info)


def test_default_is_fixed_not_worker_count():
    assert _streaming_subchunk_slot_count(_params(), _LiveDistributedClient()) == 8


def test_sync_gets_single_slot():
    assert _streaming_subchunk_slot_count(_params(), None) == 1


def test_explicit_override_wins():
    assert (
        _streaming_subchunk_slot_count(
            _params(residual_streaming_subchunks=5), _LiveDistributedClient()
        )
        == 5
    )
