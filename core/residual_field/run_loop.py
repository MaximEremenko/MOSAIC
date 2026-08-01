"""The residual-field scheduler run loop.

Extracted from ``run_residual_field_stage``, which had accreted every new
mechanism (early flush, dead-owner rescue, infra backoff, owner-epoch
fencing) as another closure over the same mutable locals. Those locals are
this class's fields and those closures are its methods — same code, same
order, now reachable from a test without standing up the whole stage.

Collaborators that unit tests monkeypatch on ``residual_field.execution``
are resolved through that module at CALL time (``_execution.<name>``), so
patching keeps working exactly as before the move.
"""
from __future__ import annotations

import logging
import time

from core.residual_field.cluster_helpers import (
    _cluster_host_memory_pressure,
    _current_worker_addresses,
    _resolve_owner_address,
    _trim_workers_for_memory_pressure,
)
from core.residual_field.contracts import ResidualFieldWorkUnit
from core.residual_field.progress_logging import (
    _format_elapsed_eta,
    _log_async_residual_progress,
    _should_log_async_progress,
    _work_unit_interval_label,
)
from core.residual_field.runtime_policy import _memory_backpressure_poll_seconds
from core.residual_field.tasks import build_residual_rifft_payload
from core.residual_field.work_unit_utils import (
    _interval_inputs_for_work_unit,
    _reducer_target_key,
    _unique_reducer_target_keys,
)
from core.runtime import is_sync_client, logging_redirect_tqdm, progress_bar

logger = logging.getLogger(__name__)


def _execution():
    """Late import of the stage module.

    Both directions of this pair are real: the stage builds the loop, and
    the loop calls stage-level collaborators that tests patch there.
    Resolving at call time keeps the import acyclic AND keeps the patch
    seams live.
    """
    from core.residual_field import execution

    return execution


class ResidualRunLoop:
    """Submission, completion, retry and rescue for one residual stage run."""

    def __init__(
        self,
        *,
        client,
        artifacts,
        chunk_futures,
        chunk_ids,
        streaming_context,
        streaming_context_future,
        task_reducer_backend,
        planned_work_units,
        work_units,
        total_tasks,
        total_reciprocal_points,
        total_partials_by_target,
        scratch_root,
        owner_local_reducer,
        worker_addresses,
        target_owners,
        nufft_resources,
        nufft_settings,
        runtime_provenance,
        reuse_rifft_payload,
        transient_interval_payloads,
        manifests_by_chunk,
        stage_task_logs,
        max_inflight,
        retries_left,
        infra_retry_budget,
        killed_worker_retry_cap,
        fail_threshold: int = 3,
    ) -> None:
        # --- run configuration (read-only for the loop) ---
        self.client = client
        self.artifacts = artifacts
        self.chunk_futures = chunk_futures
        self.chunk_ids = chunk_ids
        self.streaming_context = streaming_context
        self.task_reducer_backend = task_reducer_backend
        self.planned_work_units = planned_work_units
        self.work_units = work_units
        self.total_tasks = int(total_tasks)
        self.total_reciprocal_points = total_reciprocal_points
        self.total_partials_by_target = total_partials_by_target
        self.scratch_root = scratch_root
        self.owner_local_reducer = owner_local_reducer
        self.worker_addresses = worker_addresses
        self.target_owners = target_owners
        self.nufft_resources = nufft_resources
        self.nufft_settings = nufft_settings
        self.runtime_provenance = runtime_provenance
        self.reuse_rifft_payload = reuse_rifft_payload
        self.transient_interval_payloads = transient_interval_payloads
        self.manifests_by_chunk = manifests_by_chunk
        self.stage_task_logs = stage_task_logs
        self.infra_retry_budget = int(infra_retry_budget)
        self.killed_worker_retry_cap = int(killed_worker_retry_cap)
        self.fail_threshold = int(fail_threshold)

        # --- scheduler state ---
        self.streaming_context_future = streaming_context_future
        self.max_inflight = int(max_inflight)
        self.retries_left = retries_left
        self.infra_failures_seen: dict[tuple[str, int], int] = {}
        self.killed_worker_seen: dict[tuple[str, int], int] = {}
        # Fencing epochs per reducer target: bumped whenever ownership MOVES
        # (dead-owner rescue, retry remap). Passed into the fold task; the
        # accumulator sequences each tenure's snapshots from epoch * STRIDE
        # and the manifest union refuses to move a partition's seq backwards,
        # so a scheduler-declared-dead-but-alive predecessor cannot overwrite
        # or race the replacement's snapshots (it fails loudly at commit).
        self.target_owner_epochs: dict[tuple[int, int | None], int] = {}
        self.target_last_owner: dict[tuple[int, int | None], str] = {}
        self.deferred_resubmits: list = []  # (eligible_monotonic_time, work_unit)
        self.target_rifft_futures: dict[tuple[int, int | None], object] = {}
        self.target_remaining = {
            target_key: sum(
                1
                for work_unit in work_units
                if _reducer_target_key(work_unit) == target_key
            )
            for target_key in _unique_reducer_target_keys(work_units)
        }
        self.flying: set = set()
        self.future_meta: dict = {}
        self.future_pinned_owner: dict = {}
        self.early_flush_futures: dict[tuple[int, int | None], object] = {}
        self.exhausted_failures: list[tuple[ResidualFieldWorkUnit, str]] = []
        # Wall-clock since the cluster last had a live worker; the drain loop
        # aborts past the horizon instead of spinning on wait timeouts forever
        # (SLURM allocation revoked / node failure with the scheduler alive).
        self.dead_cluster_since: float | None = None
        self.submitted = 0
        self.completed = 0
        self.fail_streak = 0
        self.gpu_tripped = False
        self.last_cpu_trip_broadcast = 0.0
        self.last_memory_pressure_trim = 0.0
        self.last_memory_pressure_check = time.monotonic()
        self.start_time = time.monotonic()

    # ------------------------------------------------------------------ #
    #  Circuit breaker                                                    #
    # ------------------------------------------------------------------ #

    def _trip_to_cpu_only(self) -> None:
        # A worker the nanny restarts AFTER the trip comes back GPU-enabled
        # while the driver still believes gpu_tripped. Its failures rebuild
        # fail_streak past the threshold, which re-enters here — so the
        # set_cpu_only broadcast is re-sent (rate-limited) instead of
        # one-shot, pulling restarted workers back into the CPU-only regime.
        now = time.monotonic()
        if self.gpu_tripped and now - self.last_cpu_trip_broadcast < 60.0:
            return
        if hasattr(self.client, "run"):
            try:
                from core.adapters.cunufft_wrapper import set_cpu_only

                self.client.run(set_cpu_only, True)
            except Exception:
                pass
        self.last_cpu_trip_broadcast = now
        if self.gpu_tripped:
            return
        self.max_inflight = min(self.max_inflight, 256)
        self.gpu_tripped = True
        logger.warning(
            "Circuit-breaker: switching residual-field to CPU-only & throttling."
        )

    # ------------------------------------------------------------------ #
    #  Submission                                                         #
    # ------------------------------------------------------------------ #

    def _target_rifft_payload_future(
        self,
        work_unit: ResidualFieldWorkUnit,
        *,
        owner_address: str | None = None,
    ):
        if not self.reuse_rifft_payload:
            return None
        target_key = _reducer_target_key(work_unit)
        existing = self.target_rifft_futures.get(target_key)
        if existing is not None:
            return existing
        submit_kwargs = dict(
            key=(
                "riff-grid:residual:"
                f"chunk-{int(work_unit.chunk_id)}:"
                f"partition-{work_unit.partition_id if work_unit.partition_id is not None else 'owner'}"
            ),
            pure=False,
        )
        if owner_address is not None:
            submit_kwargs["workers"] = [owner_address]
            submit_kwargs["allow_other_workers"] = False
        future = self.client.submit(
            build_residual_rifft_payload,
            self.chunk_futures[int(work_unit.chunk_id)],
            work_unit=work_unit,
            quiet_logs=False,
            **submit_kwargs,
        )
        self.target_rifft_futures[target_key] = future
        return future

    def _submit_early_target_flush(self, target_key) -> None:
        """Flush a reducer target the moment its LAST work unit completes.

        The end-of-stage barrier previously wrote ~271 GB of snapshots with
        zero compute overlap (90-135 s, GPUs idle). Targets finish staggered
        (slot-interleaved submission), so streaming their final snapshots
        during the tail degrades the barrier to a short wait on writes that
        are already running. Failure here is harmless: the key stays absent
        and the barrier resubmits fresh."""
        flush_helper = _execution().flush_process_local_residual_reducer_target
        if (
            not self.owner_local_reducer
            or not self.worker_addresses
            or flush_helper is None
            or target_key not in self.target_owners
            or target_key in self.early_flush_futures
        ):
            return
        try:
            owner_address = _resolve_owner_address(
                target_key=target_key,
                target_owners=self.target_owners,
                worker_addresses=_current_worker_addresses(self.client),
            )
            if owner_address is None:
                return
            # Store the pinned owner with the future: the barrier must not
            # wait on a future whose only allowed worker has since died.
            self.early_flush_futures[target_key] = (
                self.client.submit(
                    flush_helper,
                    self.task_reducer_backend,
                    chunk_id=int(target_key[0]),
                    parameter_digest=self.planned_work_units[0].parameter_digest,
                    output_dir=self.artifacts.output_dir,
                    db_path=self.artifacts.db_manager.db_path,
                    partition_id=target_key[1],
                    pure=False,
                    workers=[owner_address],
                    allow_other_workers=False,
                ),
                owner_address,
            )
        except Exception:
            self.early_flush_futures.pop(target_key, None)

    def _mark_target_work_unit_done(
        self, work_unit: ResidualFieldWorkUnit | None
    ) -> None:
        if work_unit is None:
            return
        target_key = _reducer_target_key(work_unit)
        if target_key not in self.target_remaining:
            return
        self.target_remaining[target_key] = int(self.target_remaining[target_key]) - 1
        if self.target_remaining[target_key] > 0:
            return
        self.target_remaining.pop(target_key, None)
        if self.reuse_rifft_payload:
            rifft_future = self.target_rifft_futures.pop(target_key, None)
            release = getattr(rifft_future, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass
        self._submit_early_target_flush(target_key)

    def submit(self, work_unit: ResidualFieldWorkUnit) -> None:
        target_key = _reducer_target_key(work_unit)
        owner_address = None
        if target_key in self.target_owners:
            owner_address = _resolve_owner_address(
                target_key=target_key,
                target_owners=self.target_owners,
                worker_addresses=_current_worker_addresses(self.client),
            )
        if owner_address is not None:
            previous_owner = self.target_last_owner.get(target_key)
            if previous_owner is not None and previous_owner != owner_address:
                self.target_owner_epochs[target_key] = (
                    int(self.target_owner_epochs.get(target_key, 0)) + 1
                )
                logger.warning(
                    "Residual-field owner epoch bump | target=%s | %s -> %s "
                    "| epoch=%d",
                    target_key,
                    previous_owner,
                    owner_address,
                    self.target_owner_epochs[target_key],
                )
            self.target_last_owner[target_key] = owner_address
        submit_kwargs = dict(
            total_reciprocal_points=self.total_reciprocal_points,
            output_dir=self.artifacts.output_dir,
            db_path=(
                self.artifacts.db_manager.db_path if self.owner_local_reducer else None
            ),
            scratch_root=self.scratch_root,
            reducer_backend=self.task_reducer_backend,
            total_expected_partials=self.total_partials_by_target[target_key],
            owner_local_reducer=self.owner_local_reducer,
            quiet_logs=False,
            key=f"residual-{work_unit.artifact_key}",
            pure=False,
            resources=self.nufft_resources,
            # The driver owns retry (retries_left + infra budget + breaker).
            # Dask-level retries would run invisibly underneath it — in cpu
            # policy a deterministic failure executed up to (1+4)x(1+4)=25
            # times while fail_streak saw one failure per driver attempt.
            retries=0,
            runtime_provenance=self.runtime_provenance,
            nufft_eps=self.nufft_settings.eps,
            nufft_prefer_cpu=self.nufft_settings.prefer_cpu,
            nufft_gpu_only=self.nufft_settings.gpu_only,
            owner_epoch=int(self.target_owner_epochs.get(target_key, 0)),
        )
        if self.streaming_context_future is not None:
            submit_kwargs["streaming_compute_context"] = self.streaming_context_future
        if self.reuse_rifft_payload:
            submit_kwargs["rifft_payload"] = self._target_rifft_payload_future(
                work_unit,
                owner_address=owner_address,
            )
        if owner_address is not None:
            submit_kwargs["workers"] = [owner_address]
            submit_kwargs["allow_other_workers"] = False
        future = self.client.submit(
            _execution().run_residual_field_interval_chunk_task,
            work_unit,
            ()
            if self.streaming_context_future is not None
            else _interval_inputs_for_work_unit(
                work_unit,
                transient_interval_payloads=self.transient_interval_payloads,
            ),
            None
            if self.reuse_rifft_payload
            else self.chunk_futures[int(work_unit.chunk_id)],
            **submit_kwargs,
        )
        self.flying.add(future)
        self.future_meta[future] = work_unit
        self.future_pinned_owner[future] = owner_address
        self.submitted += 1
        if _should_log_async_progress(
            phase="queue",
            count=self.submitted,
            total=self.total_tasks,
        ):
            _log_async_residual_progress(
                enabled=self.stage_task_logs,
                event="queue",
                work_unit=work_unit,
                completed=self.completed,
                total=self.total_tasks,
                submitted=self.submitted,
                running=len(self.flying),
            )

    # ------------------------------------------------------------------ #
    #  Completion handling                                                #
    # ------------------------------------------------------------------ #

    def _incorporate_completed_result(
        self,
        future,
        completed_work_unit: ResidualFieldWorkUnit | None,
    ) -> None:
        if completed_work_unit is None:
            return
        payload = future.result()
        _execution()._record_residual_task_result(
            payload=payload,
            work_unit=completed_work_unit,
            manifests_by_chunk=self.manifests_by_chunk,
        )

    @staticmethod
    def _future_completed_successfully(future, result_marker) -> bool:
        if result_marker is False:
            return False
        status = getattr(future, "status", None)
        if status is not None:
            return status == "finished" and result_marker is not None
        return result_marker is not None and result_marker is not False

    @staticmethod
    def _future_failure_detail(future, result_marker) -> str:
        exception = None
        exception_method = getattr(future, "exception", None)
        if callable(exception_method):
            try:
                # timeout=0 through distributed's sync bridge ALWAYS raises
                # a fabricated 'timed out after 0 s' from a non-loop thread,
                # masking the real KilledWorker/Cancelled and making every
                # failure classify as infrastructure. The state is already
                # resolved locally (as_completed delivered it), so a
                # positive timeout returns on the first loop tick.
                exception = exception_method(timeout=30)
            except TypeError:
                try:
                    exception = exception_method()
                except Exception as err:
                    exception = err
            except Exception as err:
                exception = err
        if exception is not None:
            return f"{type(exception).__name__}: {exception}"
        try:
            value = future.result()
        except Exception as err:
            return f"{type(err).__name__}: {err}"
        if value is None:
            return "task returned None"
        return (
            f"task status={getattr(future, 'status', 'unknown')} "
            f"result={result_marker!r}"
        )

    @staticmethod
    def _release_finished_future(future) -> None:
        release = getattr(future, "release", None)
        if not callable(release):
            return
        try:
            release()
        except Exception:
            pass

    def _handle_completed_future(self, future, result_marker, bump, pbar=None) -> None:
        self.flying.discard(future)
        self.future_pinned_owner.pop(future, None)
        work_unit = self.future_meta.pop(future, None)
        ok = self._future_completed_successfully(future, result_marker)
        bump()
        self.completed += 1
        detail = "" if ok else self._future_failure_detail(future, result_marker)
        if work_unit is not None and pbar is not None:
            self._update_pbar_postfix(pbar, work_unit, ok=ok)
            if not ok:
                logger.warning(
                    "Residual-field batch FAILED | chunk=%d | partition=%s | intervals=%s | %s",
                    work_unit.chunk_id,
                    "owner" if work_unit.partition_id is None else work_unit.partition_id,
                    _work_unit_interval_label(work_unit),
                    detail,
                )
        if not ok and work_unit is not None:
            # Only GENUINE task failures count toward the GPU circuit
            # breaker. Infrastructure casualties — a nanny restarting a
            # worker over its memory budget, cancelled/lost futures, comm
            # drops — say nothing about GPU health, and counting them is
            # what turned every memory hiccup into a full CPU-only run
            # (measured: one 95%-budget restart cascaded into thousands of
            # cancelled batches, all "failures", breaker tripped, run dead).
            infrastructure_failure = any(
                marker in detail
                for marker in (
                    "KilledWorker",
                    "Cancelled",
                    "CommClosed",
                    "TimeoutError",
                    "WorkerProcessDied",
                    "Nanny",
                )
            )
            key = (str(work_unit.artifact_key), int(work_unit.chunk_id))
            self._release_finished_future(future)
            if infrastructure_failure:
                if "KilledWorker" in detail:
                    killed = int(self.killed_worker_seen.get(key, 0)) + 1
                    self.killed_worker_seen[key] = killed
                    if killed >= self.killed_worker_retry_cap:
                        logger.error(
                            "Residual-field unit killed its worker %d times "
                            "| chunk=%d | partition=%s | intervals=%s — "
                            "poison task, failing it instead of burning "
                            "more workers.",
                            killed,
                            work_unit.chunk_id,
                            "owner"
                            if work_unit.partition_id is None
                            else work_unit.partition_id,
                            _work_unit_interval_label(work_unit),
                        )
                        self.exhausted_failures.append((work_unit, detail))
                        self._mark_target_work_unit_done(work_unit)
                        return
                # Own budget + exponential backoff: resubmitting into the
                # middle of a worker-restart storm just dies again in
                # seconds and used to exhaust the genuine retry budget.
                used = int(self.infra_failures_seen.get(key, 0))
                self.infra_failures_seen[key] = used + 1
                if used < self.infra_retry_budget:
                    delay = min(60.0, 2.0**used)
                    self.deferred_resubmits.append(
                        (time.monotonic() + delay, work_unit)
                    )
                    logger.warning(
                        "Residual-field infra failure | chunk=%d | partition=%s "
                        "| resubmit deferred %.0fs (infra attempt %d/%d)",
                        work_unit.chunk_id,
                        "owner"
                        if work_unit.partition_id is None
                        else work_unit.partition_id,
                        delay,
                        used + 1,
                        self.infra_retry_budget,
                    )
                    return
                self.exhausted_failures.append((work_unit, detail))
                self._mark_target_work_unit_done(work_unit)
                return
            self.fail_streak += 1
            if self.fail_streak >= self.fail_threshold:
                self._trip_to_cpu_only()
            remaining = int(self.retries_left.get(key, 0))
            if remaining > 0:
                self.retries_left[key] = remaining - 1
                logger.warning(
                    "Retrying residual-field batch | chunk=%d | partition=%s | intervals=%s | remaining=%d",
                    work_unit.chunk_id,
                    "owner" if work_unit.partition_id is None else work_unit.partition_id,
                    _work_unit_interval_label(work_unit),
                    int(self.retries_left[key]),
                )
                self.submit(work_unit)
                return
            self.exhausted_failures.append((work_unit, detail))
            self._mark_target_work_unit_done(work_unit)
            return
        self.fail_streak = 0
        if work_unit is not None:
            try:
                self._incorporate_completed_result(future, work_unit)
            finally:
                self._release_finished_future(future)
                self._mark_target_work_unit_done(work_unit)

    def _process_deferred_resubmits(self) -> None:
        if not self.deferred_resubmits:
            return
        now = time.monotonic()
        ready = [item for item in self.deferred_resubmits if item[0] <= now]
        for item in ready:
            self.deferred_resubmits.remove(item)
            self.submit(item[1])

    # ------------------------------------------------------------------ #
    #  Infrastructure recovery                                            #
    # ------------------------------------------------------------------ #

    def _ensure_streaming_context_replicas(self) -> None:
        """Re-scatter the streaming compute context when every worker that
        held a replica died — dependents of a lost scattered future fail
        with immediate comm timeouts, and a restarted worker never receives
        the broadcast."""
        if self.streaming_context is None or self.streaming_context_future is None:
            return
        try:
            key = getattr(self.streaming_context_future, "key", None)
            holders = self.client.who_has(self.streaming_context_future)
            held = holders.get(key) if isinstance(holders, dict) else None
            if held:
                return
        except Exception:
            return
        try:
            logger.warning(
                "Streaming context replicas lost with dead workers; re-scattering."
            )
            self.streaming_context_future = self.client.scatter(
                self.streaming_context, broadcast=True, hash=False
            )
        except Exception:
            logger.exception("Streaming context re-scatter failed.")

    def _rescue_futures_pinned_to_dead_workers(self) -> None:
        """A queued fold task pinned (allow_other_workers=False) to a dead
        worker parks in no-worker state FOREVER — nanny restarts come back
        on NEW addresses, so nothing ever schedules it and the drain hangs
        (observed: kernel OOM killed 2 of 4 workers; 30 'running' units
        never moved again). Cancel and resubmit through the owner remap.
        Infrastructure recovery, not a task failure: no retry budget spent,
        no circuit-breaker count."""
        live = set(_current_worker_addresses(self.client))
        if not live:
            return
        self._ensure_streaming_context_replicas()
        for future in list(self.flying):
            pinned = self.future_pinned_owner.get(future)
            if pinned is None or pinned in live:
                continue
            if getattr(future, "status", "") in ("finished", "error", "cancelled"):
                continue  # completion path will handle it
            work_unit = self.future_meta.get(future)
            try:
                future.cancel()
            except Exception:
                pass
            self.flying.discard(future)
            self.future_pinned_owner.pop(future, None)
            self.future_meta.pop(future, None)
            if work_unit is None:
                continue
            target_key = _reducer_target_key(work_unit)
            # The cached rifft-payload future is pinned to the dead owner
            # too — drop it so submit() rebuilds it on the new owner.
            stale_rifft = self.target_rifft_futures.pop(target_key, None)
            release = getattr(stale_rifft, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass
            logger.warning(
                "Residual-field owner %s is gone; resubmitting chunk=%d "
                "partition=%s via owner remap",
                pinned,
                int(work_unit.chunk_id),
                "owner" if work_unit.partition_id is None else work_unit.partition_id,
            )
            self.submit(work_unit)

    # ------------------------------------------------------------------ #
    #  Draining                                                           #
    # ------------------------------------------------------------------ #

    def drain_one_completion(
        self, bump, pbar=None, timeout_seconds: float = 45.0
    ) -> bool:
        """Handle ONE completion, waiting in bounded slices so tasks pinned
        to since-dead workers get rescued instead of blocking as_completed
        forever. Returns False only when nothing is in flight (or the
        scheduler yields nothing for a completed set — anomaly)."""
        from distributed import wait as _distributed_wait

        while self.flying:
            self._process_deferred_resubmits()
            self._rescue_futures_pinned_to_dead_workers()
            if not self.flying:
                return False
            if not is_sync_client(self.client):
                horizon = _execution()._dead_cluster_horizon_seconds()
                if _current_worker_addresses(self.client):
                    self.dead_cluster_since = None
                else:
                    now = time.monotonic()
                    if self.dead_cluster_since is None:
                        self.dead_cluster_since = now
                    elif now - self.dead_cluster_since >= horizon:
                        raise RuntimeError(
                            "Residual-field drain: no live workers for "
                            f"{horizon:.0f}s with "
                            f"{len(self.flying)} batch(es) in flight."
                        )
            try:
                _distributed_wait(
                    list(self.flying),
                    timeout=timeout_seconds,
                    return_when="FIRST_COMPLETED",
                )
            except TimeoutError:
                continue
            except Exception:
                # Sync/test clients (or comm hiccups) can make wait() raise
                # immediately — fall through to the blocking generator path
                # rather than spinning on retry.
                pass
            for future, result in _execution().yield_futures_with_results(
                list(self.flying), self.client
            ):
                self._handle_completed_future(future, result, bump, pbar=pbar)
                return True
            return False
        return False

    def _harvest_finished_nonblocking(self, bump, pbar=None) -> None:
        done_now = [future for future in list(self.flying) if future.done()]
        for future in done_now:
            try:
                result_marker = future.result()
            except Exception:
                result_marker = False
            self._handle_completed_future(future, result_marker, bump, pbar=pbar)

    def _apply_memory_backpressure(self, bump, pbar=None) -> None:
        poll_seconds = _memory_backpressure_poll_seconds()
        if poll_seconds <= 0.0:
            return
        now = time.monotonic()
        if now - self.last_memory_pressure_check < poll_seconds:
            return
        self.last_memory_pressure_check = now
        if not _cluster_host_memory_pressure(self.client):
            return
        if now - self.last_memory_pressure_trim >= 5.0:
            _trim_workers_for_memory_pressure(self.client)
            self.last_memory_pressure_trim = now
            logger.warning(
                "Residual-field memory backpressure: worker RSS is near Dask pause threshold; "
                "trimming native pools and waiting for in-flight work before submitting more batches."
            )
        while self.flying and _cluster_host_memory_pressure(self.client):
            if not self.drain_one_completion(bump, pbar=pbar):
                break

    # ------------------------------------------------------------------ #
    #  Progress + driving                                                 #
    # ------------------------------------------------------------------ #

    def _update_pbar_postfix(self, pbar, work_unit, ok=True):
        elapsed = time.monotonic() - self.start_time
        timing = _format_elapsed_eta(elapsed, self.completed, self.total_tasks)
        partition = (
            "owner"
            if work_unit.partition_id is None
            else f"p{int(work_unit.partition_id)}"
        )
        status = "" if ok else " | FAILED"
        fails = (
            f" | fails={len(self.exhausted_failures)}" if self.exhausted_failures else ""
        )
        pbar.set_postfix_str(
            f"done=chunk{int(work_unit.chunk_id)}/{partition}"
            f":{_work_unit_interval_label(work_unit)}"
            f" | running={len(self.flying)} | {timing}{fails}{status}"
        )

    def run(self) -> None:
        """Submit every work unit and drain to quiescence."""
        with logging_redirect_tqdm():
            with progress_bar(
                self.total_tasks, desc="Residual-field", unit="batch", force=True
            ) as pbar:

                def bump() -> None:
                    pbar.update(1)

                for work_unit in self.work_units:
                    self.submit(work_unit)
                    self._harvest_finished_nonblocking(bump, pbar=pbar)
                    self._apply_memory_backpressure(bump, pbar=pbar)
                    while len(self.flying) >= self.max_inflight:
                        # Drain ONE completion then break so the outer
                        # submit-loop can enqueue the next work_unit
                        # immediately; bounded waits inside keep dead-owner
                        # rescue running.
                        if not self.drain_one_completion(bump, pbar=pbar):
                            break

                while self.flying or self.deferred_resubmits:
                    self._process_deferred_resubmits()
                    if not self.flying:
                        # Everything in flight is waiting out an infra backoff.
                        time.sleep(0.5)
                        continue
                    if not self.drain_one_completion(bump, pbar=pbar) and self.flying:
                        raise RuntimeError(
                            "Residual-field scheduler made no progress while draining "
                            f"{len(self.flying)} in-flight batch(es)."
                        )

    def raise_if_batches_failed(self) -> None:
        if not self.exhausted_failures:
            return
        formatted = "; ".join(
            self._format_failed_batch(work_unit, detail)
            for work_unit, detail in self.exhausted_failures
        )
        raise RuntimeError(
            f"Residual-field batch failed after retries before finalize: {formatted}"
        )

    @staticmethod
    def _format_failed_batch(work_unit: ResidualFieldWorkUnit, detail: str) -> str:
        partition = (
            "owner"
            if work_unit.partition_id is None
            else str(int(work_unit.partition_id))
        )
        return (
            f"chunk={int(work_unit.chunk_id)} "
            f"partition={partition} "
            f"intervals={_work_unit_interval_label(work_unit)} "
            f"reason={detail}"
        )


__all__ = ["ResidualRunLoop"]
