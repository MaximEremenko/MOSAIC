from __future__ import annotations

from types import SimpleNamespace

import core.runtime.dask_helpers as dask_helpers


class _FakeFuture:
    def __init__(self, status: str) -> None:
        self.status = status


def test_yield_futures_with_results_treats_cancelled_and_error_as_failed(
    monkeypatch,
) -> None:
    finished = _FakeFuture("finished")
    cancelled = _FakeFuture("cancelled")
    errored = _FakeFuture("error")
    captured_kwargs: dict[str, object] = {}

    def fake_as_completed(futs, **kwargs):
        captured_kwargs.update(kwargs)
        assert futs == [finished, cancelled, errored]
        return iter(
            [
                (finished, {"manifest": 1}),
                (cancelled, RuntimeError("cancelled")),
                (errored, (RuntimeError, RuntimeError("boom"), None)),
            ]
        )

    monkeypatch.setattr(dask_helpers, "as_completed", fake_as_completed)

    results = list(
        dask_helpers.yield_futures_with_results(
            [finished, cancelled, errored],
            SimpleNamespace(loop="event-loop"),
        )
    )

    assert results == [
        (finished, True),
        (cancelled, False),
        (errored, False),
    ]
    assert captured_kwargs == {
        "with_results": True,
        "raise_errors": False,
        "loop": "event-loop",
    }


def test_yield_futures_with_results_treats_none_payload_as_failed(monkeypatch) -> None:
    finished = _FakeFuture("finished")

    def fake_as_completed(futs, **kwargs):
        assert futs == [finished]
        return iter([(finished, None)])

    monkeypatch.setattr(dask_helpers, "as_completed", fake_as_completed)

    assert list(dask_helpers.yield_futures_with_results([finished], None)) == [
        (finished, False)
    ]
