from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.storage.attempt_store import run_root
from core.storage.atomic import assert_path_contained
from core.storage.fingerprint import file_sha256
from core.storage.fs_capability import FSCapabilityManifest, write_fs_capability_manifest


logger = logging.getLogger(__name__)


class FilesystemCapabilityError(RuntimeError):
    """Raised when an output filesystem cannot satisfy current-run requirements."""


@dataclass(frozen=True)
class VisibilityProbeResult:
    host: str
    ok: bool
    file_sha256: str | None
    error: str | None = None


def _filesystem_type(path: Path) -> str | None:
    try:
        resolved = path.resolve()
        best_mount = Path("/")
        best_type: str | None = None
        with Path("/proc/mounts").open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = Path(parts[1].replace("\\040", " "))
                try:
                    resolved.relative_to(mount_point)
                except ValueError:
                    continue
                if len(mount_point.parts) >= len(best_mount.parts):
                    best_mount = mount_point
                    best_type = parts[2]
        return best_type
    except Exception:
        return None


def _write_probe_file(temp_path: Path, payload: bytes) -> None:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _try_directory_fsync(path: Path) -> tuple[bool, str | None]:
    try:
        fd = os.open(path, os.O_RDONLY)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    try:
        os.fsync(fd)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        os.close(fd)


def _poll_visible_hash(path: Path, expected_hash: str, *, timeout_seconds: float) -> float:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    start = time.monotonic()
    last_error: Exception | None = None
    while True:
        try:
            if path.exists() and file_sha256(path) == expected_hash:
                return time.monotonic() - start
        except Exception as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            detail = f": {last_error}" if last_error is not None else ""
            raise FilesystemCapabilityError(
                f"Renamed probe file was not visible with the expected hash{detail}."
            )
        time.sleep(0.01)


def _probe_symlink_policy(probe_dir: Path, final_path: Path) -> dict[str, Any]:
    link_path = probe_dir / "probe_link"
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(final_path.name)
        target = os.readlink(link_path)
        return {"symlink_create_allowed": True, "symlink_target": target}
    except Exception as exc:
        return {
            "symlink_create_allowed": False,
            "symlink_error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        try:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
        except OSError:
            pass


def _read_probe_file(path_text: str, expected_hash: str) -> dict[str, Any]:
    path = Path(path_text)
    host = socket.gethostname()
    try:
        observed = file_sha256(path)
        return {
            "host": host,
            "ok": observed == expected_hash,
            "file_sha256": observed,
            "error": None if observed == expected_hash else "hash mismatch",
        }
    except Exception as exc:
        return {
            "host": host,
            "ok": False,
            "file_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _client_worker_hosts(client) -> tuple[str, ...] | None:
    """Hosts this run spans, or None when that cannot be determined.

    None is NOT "one host". This value decides whether the fail-closed
    cross-host and file-lock guards run at all, so a transient client.run()
    failure that returned the local hostname silently downgraded a
    multi-node run to single-host and skipped both."""
    if client is None or not hasattr(client, "run"):
        return (socket.gethostname(),)
    try:
        results = client.run(lambda: socket.gethostname())
    except Exception as exc:
        logger.warning(
            "Could not enumerate worker hosts (%s: %s); treating the run as "
            "multi-host so the shared-filesystem guards still apply.",
            type(exc).__name__,
            exc,
        )
        return None
    if isinstance(results, Mapping):
        hosts = [str(value) for value in results.values()]
    else:
        hosts = [str(results)]
    hosts.append(socket.gethostname())
    return tuple(sorted(set(hosts)))


def cross_host_read_after_rename_probe(
    *,
    path: str | Path,
    expected_hash: str,
    client=None,
) -> tuple[VisibilityProbeResult, ...]:
    if client is None or not hasattr(client, "run"):
        payload = _read_probe_file(str(path), str(expected_hash))
        return (
            VisibilityProbeResult(
                host=str(payload["host"]),
                ok=bool(payload["ok"]),
                file_sha256=payload.get("file_sha256"),
                error=payload.get("error"),
            ),
        )
    try:
        raw = client.run(_read_probe_file, str(path), str(expected_hash))
    except Exception as exc:
        return (
            VisibilityProbeResult(
                host=socket.gethostname(),
                ok=False,
                file_sha256=None,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
    values = raw.values() if isinstance(raw, Mapping) else (raw,)
    return tuple(
        VisibilityProbeResult(
            host=str(item.get("host", "unknown")),
            ok=bool(item.get("ok")),
            file_sha256=item.get("file_sha256"),
            error=item.get("error"),
        )
        for item in values
        if isinstance(item, Mapping)
    )


def _probe_file_lock(dir_text: str) -> dict[str, Any]:
    """Runs on driver or worker: can this process take an fcntl lock on the
    shared filesystem?

    Catches a dead lockd and mounts where flock is unsupported, i.e. the
    cases that RAISE. It does NOT catch NFS mounted -o nolock or
    local_lock=all, where the kernel satisfies the lock locally and this
    probe passes while the lock coordinates nothing -- every prober locks a
    file private to itself, so there is nothing here to be excluded from.
    cross_host_lock_exclusion_probe is the test with that power. The probe
    file is PER PROCESS: all probers
    run concurrently via client.run, and a shared path turns a sibling's
    perfectly functional lock into a spurious EAGAIN (observed in the
    cluster sim: node2's 'failure' was node1 holding the probe lock —
    evidence lockd WORKS, misread as it being broken)."""
    host = socket.gethostname()
    try:
        import fcntl

        path = Path(dir_text) / f"lock_probe.{host}.{os.getpid()}.dat"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        path.unlink(missing_ok=True)
        return {"host": host, "ok": True, "error": None}
    except Exception as exc:
        return {"host": host, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


_LOCK_EXCLUSION_PROBE_NAME = "lock_exclusion.dat"


def _probe_lock_is_excluded(path_text: str) -> dict[str, Any]:
    """Try to take a lock the driver is HOLDING. Failing is the pass."""
    host = socket.gethostname()
    try:
        import fcntl

        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return {"host": host, "acquired": False, "error": None}
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return {"host": host, "acquired": True, "error": None}
    except Exception as exc:
        return {
            "host": host,
            "acquired": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def cross_host_lock_exclusion_probe(
    *,
    probe_dir: str | Path,
    client=None,
) -> tuple[dict[str, Any], ...]:
    """Does a held lock actually EXCLUDE anyone else?

    :func:`_probe_file_lock` cannot answer this: every prober locks a file
    private to itself, so the only thing that can make it fail is ENOLCK or
    an unsupported mount. On NFS mounted ``-o nolock`` or ``local_lock=all``
    the kernel satisfies flock locally and the probe passes while the lock
    coordinates nothing — exactly the mounts it claims to catch.

    So hold a lock on a SHARED file here and ask every worker to take the
    same lock non-blocking. Each one must FAIL. Any worker that acquires it
    while we hold it proves the lock does not exclude, and the chunk mutex
    guarding the reducer-progress read-modify-write is decorative."""
    import fcntl

    probe_path = Path(probe_dir) / _LOCK_EXCLUSION_PROBE_NAME
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    try:
        with probe_path.open("a+b") as holder:
            try:
                fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                return (
                    {
                        "host": socket.gethostname(),
                        "ok": False,
                        "error": f"driver could not take the probe lock: {exc}",
                    },
                )
            try:
                if client is not None and hasattr(client, "run"):
                    raw = client.run(_probe_lock_is_excluded, str(probe_path))
                    values = raw.values() if isinstance(raw, Mapping) else (raw,)
                    for item in values:
                        if not isinstance(item, Mapping):
                            continue
                        results.append(
                            {
                                "host": str(item.get("host", "unknown")),
                                "ok": not bool(item.get("acquired")),
                                "error": (
                                    "acquired a lock the driver was holding"
                                    if item.get("acquired")
                                    else item.get("error")
                                ),
                            }
                        )
            finally:
                fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        return (
            {
                "host": socket.gethostname(),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
    finally:
        probe_path.unlink(missing_ok=True)
    return tuple(results)


def cross_host_file_lock_probe(
    *,
    probe_dir: str | Path,
    client=None,
) -> tuple[dict[str, Any], ...]:
    results: list[dict[str, Any]] = [_probe_file_lock(str(probe_dir))]
    if client is not None and hasattr(client, "run"):
        try:
            raw = client.run(_probe_file_lock, str(probe_dir))
        except Exception as exc:
            results.append(
                {
                    "host": "workers",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            values = raw.values() if isinstance(raw, Mapping) else (raw,)
            results.extend(item for item in values if isinstance(item, Mapping))
    deduped: dict[str, dict[str, Any]] = {}
    for item in results:
        host = str(item.get("host", "unknown"))
        # a failing probe for a host wins over a passing one (two workers on
        # one host can disagree only transiently; fail closed)
        if host not in deduped or not item.get("ok"):
            deduped[host] = dict(item)
    return tuple(deduped[host] for host in sorted(deduped))


def _capability_failure_message(reason: str) -> str:
    return (
        f"Output filesystem capability check failed: {reason}. "
        "Use single-host mode, choose a different shared filesystem, adjust "
        "NFS mount/cache settings, run with manifest-only DB mode, or rerun "
        "the filesystem capability profile after fixing the environment."
    )


def profile_output_filesystem(
    output_dir: str | Path,
    *,
    run_digest: str,
    client=None,
    require_cross_host: bool = False,
    visibility_timeout_seconds: float = 5.0,
) -> FSCapabilityManifest:
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root(output_root, run_digest)
    probe_dir = assert_path_contained(
        run_dir / "runtime_probes" / "fs_capability",
        output_dir=output_root,
    )
    probe_dir.mkdir(parents=True, exist_ok=True)

    payload = f"mosaic-fs-capability:{run_digest}:{time.time_ns()}".encode("utf-8")
    temp_path = probe_dir / ".rename_probe.tmp"
    final_path = probe_dir / "rename_probe.dat"
    try:
        if temp_path.exists():
            temp_path.unlink()
        if final_path.exists():
            final_path.unlink()
        _write_probe_file(temp_path, payload)
        expected_hash = file_sha256(temp_path)
        replace_start = time.monotonic()
        os.replace(temp_path, final_path)
        replace_latency = time.monotonic() - replace_start
        directory_fsync_supported, directory_fsync_error = _try_directory_fsync(
            final_path.parent
        )
        visibility_latency = _poll_visible_hash(
            final_path,
            expected_hash,
            timeout_seconds=visibility_timeout_seconds,
        )
        hosts = _client_worker_hosts(client)
        # Unknown topology counts as multi-host: the guards below are the
        # fail-closed ones, and skipping them because we could not ask is
        # the failure this treats as a failure.
        needs_cross_host = bool(
            require_cross_host or hosts is None or len(hosts) > 1
        )
        cross_host_results = cross_host_read_after_rename_probe(
            path=final_path,
            expected_hash=expected_hash,
            client=client,
        )
        cross_host_ok = all(result.ok for result in cross_host_results)
        if needs_cross_host and not cross_host_ok:
            failures = ", ".join(
                f"{result.host}:{result.error or result.file_sha256}"
                for result in cross_host_results
                if not result.ok
            )
            raise FilesystemCapabilityError(
                _capability_failure_message(f"cross-host read-after-rename failed ({failures})")
            )
        # File locking is the one capability the multi-node design actually
        # depends on (chunk-mutex exclusion of the reducer-progress
        # manifest read-modify-write), and the one this profile never used
        # to test. Fail closed for multi-node runs; single-host runs only
        # record the result (the in-process thread lock suffices there).
        lock_probe_results = cross_host_file_lock_probe(
            probe_dir=probe_dir,
            client=client,
        )
        lock_ok = all(bool(item.get("ok")) for item in lock_probe_results)
        if needs_cross_host and not lock_ok:
            failures = ", ".join(
                f"{item.get('host')}:{item.get('error')}"
                for item in lock_probe_results
                if not item.get("ok")
            )
            raise FilesystemCapabilityError(
                _capability_failure_message(
                    f"file locking unavailable on shared filesystem ({failures})"
                )
            )
        # Taking a lock is not the capability we need; EXCLUDING someone
        # else is. The probe above cannot tell them apart.
        lock_exclusion_results = cross_host_lock_exclusion_probe(
            probe_dir=probe_dir,
            client=client,
        )
        lock_exclusion_ok = all(
            bool(item.get("ok")) for item in lock_exclusion_results
        )
        if needs_cross_host and lock_exclusion_results and not lock_exclusion_ok:
            failures = ", ".join(
                f"{item.get('host')}:{item.get('error')}"
                for item in lock_exclusion_results
                if not item.get("ok")
            )
            raise FilesystemCapabilityError(
                _capability_failure_message(
                    "file locks do not exclude across the shared filesystem "
                    f"({failures}); a nolock / local_lock=all mount satisfies "
                    "flock locally while coordinating nothing"
                )
            )
        capabilities: dict[str, Any] = {
            "filesystem_type": _filesystem_type(output_root),
            "same_directory_atomic_replace_visible": True,
            "same_directory_replace_latency_seconds": float(replace_latency),
            "reopen_after_rename_sha256": expected_hash,
            "directory_fsync_supported": bool(directory_fsync_supported),
            "directory_fsync_error": directory_fsync_error,
            "symlink_policy": _probe_symlink_policy(probe_dir, final_path),
            "max_metadata_visibility_latency_seconds": float(visibility_latency),
            "driver_host": socket.gethostname(),
            "worker_hosts": None if hosts is None else list(hosts),
            "cross_host_required": bool(needs_cross_host),
            "cross_host_read_after_rename": [
                {
                    "host": result.host,
                    "ok": result.ok,
                    "file_sha256": result.file_sha256,
                    "error": result.error,
                }
                for result in cross_host_results
            ],
            "hardlink_required": False,
            "file_lock_required": bool(needs_cross_host),
            "file_lock_functional": bool(lock_ok),
            "file_lock_per_host": list(lock_probe_results),
            "file_lock_excludes_across_hosts": bool(lock_exclusion_ok),
            "file_lock_exclusion_per_host": list(lock_exclusion_results),
            "symlink_required": False,
        }
        return write_fs_capability_manifest(
            output_dir=output_root,
            run_digest=run_digest,
            capabilities=capabilities,
        )
    except FilesystemCapabilityError:
        raise
    except Exception as exc:
        raise FilesystemCapabilityError(
            _capability_failure_message(f"{type(exc).__name__}: {exc}")
        ) from exc
    finally:
        for candidate in (temp_path, final_path):
            try:
                if candidate.exists():
                    candidate.unlink()
            except OSError:
                pass


__all__ = [
    "FilesystemCapabilityError",
    "VisibilityProbeResult",
    "cross_host_read_after_rename_probe",
    "profile_output_filesystem",
]
