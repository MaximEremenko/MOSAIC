from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.runtime.dask_helpers import yield_futures_with_results
from core.storage.attempt_store import work_unit_attempts_root


@dataclass(frozen=True)
class QuiescenceReport:
    stage: str
    chunk_id: int
    terminal_futures: int
    failed_futures: int
    discovered_attempt_manifests: int
    missing_work_unit_digests: tuple[str, ...]


def _future_done(future) -> bool:
    done = getattr(future, "done", None)
    if callable(done):
        try:
            return bool(done())
        except Exception:
            return True
    status = getattr(future, "status", None)
    return status in {"finished", "error", "cancelled"}


def _future_failed(future) -> bool:
    status = getattr(future, "status", None)
    if status in {"error", "cancelled"}:
        return True
    if status == "finished":
        return False
    exception = getattr(future, "exception", None)
    if callable(exception):
        try:
            return exception() is not None
        except Exception:
            return True
    return False


def _attempt_manifest_count(
    *,
    output_dir: str | Path,
    run_digest: str,
    stage: str,
    chunk_id: int,
    work_unit_digest: str,
) -> int:
    root = work_unit_attempts_root(
        output_dir,
        run_digest,
        stage,
        int(chunk_id),
        str(work_unit_digest),
    )
    if not root.exists():
        return 0
    return len(sorted(root.glob("attempt_*/attempt.json")))


def require_chunk_quiescence(
    futures: Iterable[object],
    *,
    client,
    output_dir: str | Path,
    run_digest: str,
    stage: str,
    chunk_id: int,
    expected_work_unit_digests: Iterable[str],
) -> QuiescenceReport:
    future_list = list(futures)
    pending = [future for future in future_list if not _future_done(future)]
    if pending:
        for _future, _result in yield_futures_with_results(pending, client):
            pass
    still_pending = [future for future in future_list if not _future_done(future)]
    if still_pending:
        raise RuntimeError(
            f"{stage} chunk {int(chunk_id)} did not reach quiescence: "
            f"{len(still_pending)} future(s) are not terminal."
        )
    failed = [future for future in future_list if _future_failed(future)]
    if failed:
        raise RuntimeError(
            f"{stage} chunk {int(chunk_id)} reached quiescence with "
            f"{len(failed)} failed future(s); refusing commit candidate creation."
        )

    missing: list[str] = []
    discovered = 0
    for digest in sorted(str(item) for item in expected_work_unit_digests):
        count = _attempt_manifest_count(
            output_dir=output_dir,
            run_digest=run_digest,
            stage=stage,
            chunk_id=int(chunk_id),
            work_unit_digest=digest,
        )
        discovered += int(count)
        if count <= 0:
            missing.append(digest)
    if missing:
        raise RuntimeError(
            f"{stage} chunk {int(chunk_id)} missing attempt manifests after "
            f"quiescence for work units: {', '.join(missing)}"
        )
    return QuiescenceReport(
        stage=str(stage),
        chunk_id=int(chunk_id),
        terminal_futures=len(future_list),
        failed_futures=len(failed),
        discovered_attempt_manifests=int(discovered),
        missing_work_unit_digests=tuple(missing),
    )


__all__ = ["QuiescenceReport", "require_chunk_quiescence"]
