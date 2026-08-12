"""Durable snapshot family validation and chunk-payload assembly.

Pure functions extracted from the reducer backend god-module: they take
snapshot metadata + paths and return an assembled chunk payload, with no
backend state involved.
"""
from __future__ import annotations

import logging

import numpy as np

from core.residual_field.local_accumulator import load_local_accumulator_snapshot
from core.residual_field.reducer_helpers import _allocate_finalize_output

logger = logging.getLogger(__name__)


def validate_local_partition_snapshot_family(
    snapshot_metadata,
    *,
    chunk_id,
) -> None:
    """Reject checkpoint families that cannot assemble into a correct chunk.

    Partition snapshots are only safe to concatenate when (a) every partition
    incorporated the same interval set — partitions split real-space points,
    so a differing set means at least one partition is missing contributions —
    and (b) their atom ranges tile ``[0, n_atoms)`` contiguously in assembly
    (partition-id) order. Stale snapshots surviving from an older partition
    layout (crash + code/flag change + resume) violate one of these and used
    to be concatenated silently, publishing a truncated or corrupt chunk.
    """
    entries = [
        (partition_id, metadata)
        for partition_id, _seq, metadata in snapshot_metadata
        if metadata is not None
    ]
    if len(entries) <= 1:
        return
    if any(partition_id is None for partition_id, _metadata in entries):
        raise RuntimeError(
            "Residual-field local finalization found a mix of owner-level and "
            f"partitioned checkpoints for chunk={int(chunk_id)}. The checkpoint "
            "family is inconsistent (stale snapshots from a different partition "
            "layout); delete 'residual_checkpoints/' under the output directory "
            "and re-run to recompute this chunk."
        )
    interval_sets = {
        frozenset(int(v) for v in metadata["incorporated_interval_ids"])
        for _partition_id, metadata in entries
    }
    if len(interval_sets) > 1:
        sizes = sorted(len(s) for s in interval_sets)
        raise RuntimeError(
            "Residual-field local finalization found partition checkpoints with "
            f"mismatched interval coverage for chunk={int(chunk_id)} "
            f"(interval-set sizes {sizes}). At least one partition checkpoint is "
            "stale or incomplete; delete 'residual_checkpoints/' under the output "
            "directory and re-run to recompute this chunk."
        )
    ranges = [
        (metadata.get("point_start"), metadata.get("point_stop"))
        for _partition_id, metadata in entries
    ]
    if any(start is None or stop is None for start, stop in ranges):
        logger.warning(
            "Residual-field local finalization cannot verify partition tiling for "
            "chunk=%d: legacy checkpoint(s) lack point ranges.",
            int(chunk_id),
        )
        return
    expected_start = 0
    for (partition_id, _metadata), (start, stop) in zip(entries, ranges):
        if int(start) != expected_start or int(stop) <= int(start):
            raise RuntimeError(
                "Residual-field local finalization found partition checkpoints "
                f"that do not tile the chunk for chunk={int(chunk_id)}: partition "
                f"{partition_id} covers atoms [{start}:{stop}] but assembly "
                f"expected the next range to start at {expected_start}. Stale "
                "checkpoints from a different partition layout are present; "
                "delete 'residual_checkpoints/' under the output directory and "
                "re-run to recompute this chunk."
            )
        expected_start = int(stop)


def _require_expected_partition_family(
    snapshot_metadata,
    *,
    expected_partitions,
    chunk_id,
) -> None:
    """Verify the snapshot family matches the plan's partition family exactly.

    The intra-family checks in ``validate_local_partition_snapshot_family``
    cannot see a missing TAIL partition (a K-of-N family that tiles
    contiguously from 0 still passes). Only the plan knows the full family, so
    the finalize paths that run after coverage validation pass it down here.
    """
    found_ids = {
        partition_id for partition_id, _seq, metadata in snapshot_metadata
        if metadata is not None
    }
    expected_map = {
        (int(partition_id) if partition_id is not None else None): (start, stop)
        for partition_id, start, stop in expected_partitions
    }
    if found_ids != set(expected_map):
        missing = sorted(str(v) for v in set(expected_map) - found_ids)
        extra = sorted(str(v) for v in found_ids - set(expected_map))
        raise RuntimeError(
            "Residual-field local finalization found a partition-snapshot "
            f"family that does not match the plan for chunk={int(chunk_id)}: "
            f"missing partitions {missing}, unexpected partitions {extra}. "
            "Delete 'residual_checkpoints/' under the output directory and "
            "re-run to recompute this chunk."
        )
    for partition_id, _seq, metadata in snapshot_metadata:
        if metadata is None:
            continue
        snap_start = metadata.get("point_start")
        snap_stop = metadata.get("point_stop")
        if snap_start is None and snap_stop is None:
            continue
        if (snap_start, snap_stop) != expected_map[partition_id]:
            raise RuntimeError(
                "Residual-field local finalization found partition "
                f"{partition_id} of chunk={int(chunk_id)} covering atoms "
                f"[{snap_start}:{snap_stop}] where the plan expects "
                f"{expected_map[partition_id]}. Delete 'residual_checkpoints/' "
                "under the output directory and re-run to recompute this chunk."
            )


def validate_interval_partitioned_snapshot_family(
    snapshot_metadata,
    *,
    chunk_id,
) -> None:
    """Reject subchunk (interval-partitioned) families that cannot merge into
    a correct chunk.

    Subchunks are the transpose of atom partitions: each covers the chunk's
    FULL real-space extent but only a SUBSET of the hkl intervals, and the
    final chunk is their elementwise sum. That sum is only correct when
    (a) the incorporated interval sets are pairwise disjoint — an interval
    folded into two subchunks is double-counted in the merge, silently
    corrupting every amplitude it touches — and (b) every subchunk covers the
    same point range. Note the deliberate inversion of
    ``validate_local_partition_snapshot_family``: there, differing interval
    sets are the corruption signal; here, differing (disjoint) interval sets
    are the design and OVERLAP is the corruption signal.
    """
    entries = [
        (subchunk_id, metadata)
        for subchunk_id, _seq, metadata in snapshot_metadata
        if metadata is not None
    ]
    if not entries:
        return
    if len(entries) > 1 and any(
        subchunk_id is None for subchunk_id, _metadata in entries
    ):
        raise RuntimeError(
            "Residual-field streaming finalization found a mix of owner-level "
            f"and subchunk checkpoints for chunk={int(chunk_id)}. The "
            "checkpoint family is inconsistent (stale snapshots from a "
            "different reducer layout); delete 'residual_checkpoints/' under "
            "the output directory and re-run to recompute this chunk."
        )
    seen: dict[int, object] = {}
    for subchunk_id, metadata in entries:
        interval_ids = [
            int(v) for v in metadata.get("incorporated_interval_ids") or ()
        ]
        interval_set = set(interval_ids)
        if len(interval_set) != len(interval_ids):
            raise RuntimeError(
                "Residual-field streaming finalization found subchunk "
                f"{subchunk_id} of chunk={int(chunk_id)} listing duplicate "
                "interval ids in one checkpoint. The checkpoint is corrupt; "
                "delete 'residual_checkpoints/' under the output directory "
                "and re-run to recompute this chunk."
            )
        for interval_id in interval_set:
            if interval_id in seen:
                raise RuntimeError(
                    "Residual-field streaming finalization found interval "
                    f"{interval_id} of chunk={int(chunk_id)} incorporated by "
                    f"both subchunk {seen[interval_id]} and subchunk "
                    f"{subchunk_id}. Summing them would double-count that "
                    "interval's contribution; delete 'residual_checkpoints/' "
                    "under the output directory and re-run to recompute this "
                    "chunk."
                )
            seen[interval_id] = subchunk_id
    ranges = {
        (
            None if metadata.get("point_start") is None else int(metadata["point_start"]),
            None if metadata.get("point_stop") is None else int(metadata["point_stop"]),
        )
        for _subchunk_id, metadata in entries
    }
    if len(ranges) > 1:
        raise RuntimeError(
            "Residual-field streaming finalization found subchunk checkpoints "
            f"with differing point ranges for chunk={int(chunk_id)} "
            f"({sorted(ranges)}). Every subchunk must cover the chunk's full "
            "extent; stale checkpoints from a different layout are present. "
            "Delete 'residual_checkpoints/' under the output directory and "
            "re-run to recompute this chunk."
        )


def _require_expected_interval_coverage(
    snapshot_metadata,
    *,
    expected_interval_ids,
    chunk_id,
) -> None:
    """Verify the subchunk family's interval union matches the plan exactly.

    Disjointness (``validate_interval_partitioned_snapshot_family``) cannot
    see a missing interval — a K-of-N union is still disjoint. Only the plan
    knows the full interval set, so finalize passes it down here. Unknown ids
    are rejected too: they mean snapshots from an older interval plan
    survived, and their contributions would not correspond to any current
    interval."""
    union: set[int] = set()
    for _subchunk_id, _seq, metadata in snapshot_metadata:
        if metadata is None:
            continue
        union.update(
            int(v) for v in metadata.get("incorporated_interval_ids") or ()
        )
    expected = {int(v) for v in expected_interval_ids}
    if union == expected:
        return
    missing = sorted(expected - union)
    extra = sorted(union - expected)
    raise RuntimeError(
        "Residual-field streaming finalization found subchunk checkpoints "
        f"whose interval union does not match the plan for chunk={int(chunk_id)}: "
        f"{len(missing)} missing interval(s){' (e.g. ' + ', '.join(str(v) for v in missing[:5]) + ')' if missing else ''}, "
        f"{len(extra)} unknown interval(s){' (e.g. ' + ', '.join(str(v) for v in extra[:5]) + ')' if extra else ''}. "
        "Delete 'residual_checkpoints/' under the output directory and re-run "
        "to recompute this chunk."
    )


def assemble_local_snapshot_chunk_payload(
    *,
    snapshot_metadata,
    chunk_id,
    parameter_digest,
    output_dir,
    scratch_dir,
) -> dict | None:
    validate_local_partition_snapshot_family(
        snapshot_metadata,
        chunk_id=chunk_id,
    )
    # Pass 1: load one snapshot at a time to inspect shape, then
    # drop it. This keeps peak finalize RAM near one partition plus
    # the output buffers instead of all partitions at once.
    # Match np.vstack semantics: 1-D block of shape (k,) counts
    # as 1 row × k columns; 2-D block (r, c) is r rows × c cols.
    per_snapshot_sizes: list[tuple[int | None, int, int, int]] = []
    total_points = 0
    total_rows = 0
    grid_cols: int | None = None
    total_reciprocal_points: int | None = None
    for partition_id, snapshot_seq, _metadata in snapshot_metadata:
        snapshot = load_local_accumulator_snapshot(
            output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            partition_id=partition_id,
            snapshot_seq=snapshot_seq,
        )
        if snapshot is None:
            # A partition whose METADATA loaded but whose payload will not
            # is not an absent partition — it is a partition this chunk is
            # made of. Skipping it here assembled a chunk missing that
            # partition's points, published it COMMITTED, and then unlinked
            # the snapshot, so the loss became permanent and surfaced far
            # downstream as an opaque shape error. Fail here instead: the
            # snapshot is still on disk and the chunk can be recomputed.
            raise RuntimeError(
                f"Residual chunk {int(chunk_id)} partition "
                f"{'owner' if partition_id is None else int(partition_id)} "
                f"snapshot seq={int(snapshot_seq)} is listed in the progress "
                "manifest but its payload could not be loaded. Refusing to "
                "publish a chunk assembled from the remaining partitions."
            )
        n_points = int(
            np.asarray(snapshot["amplitudes_delta"]).reshape(-1).shape[0]
        )
        grid_arr = np.asarray(snapshot["grid_shape_nd"])
        if grid_arr.ndim == 1:
            block_rows = 1
            block_cols = int(grid_arr.shape[0])
        elif grid_arr.ndim >= 2:
            block_rows = int(grid_arr.shape[0])
            block_cols = int(grid_arr.shape[1])
        else:
            block_rows = 0
            block_cols = 1
        if grid_cols is None:
            grid_cols = block_cols
        per_snapshot_sizes.append(
            (partition_id, int(snapshot_seq), n_points, block_rows)
        )
        total_points += n_points
        total_rows += block_rows
        if total_reciprocal_points is None:
            total_reciprocal_points = int(snapshot["total_reciprocal_points"])
        del snapshot, grid_arr
    if not per_snapshot_sizes:
        return None
    # np.vstack always returns 2-D; mirror that unconditionally.
    grid_shape = (total_rows, int(grid_cols or 1))
    final_point_ids = _allocate_finalize_output(
        shape=(total_points,), dtype=np.int64,
        scratch_dir=scratch_dir, name="point_ids",
    )
    final_delta = _allocate_finalize_output(
        shape=(total_points,), dtype=np.complex128,
        scratch_dir=scratch_dir, name="delta",
    )
    final_average = _allocate_finalize_output(
        shape=(total_points,), dtype=np.complex128,
        scratch_dir=scratch_dir, name="average",
    )
    final_grid_shape_nd = _allocate_finalize_output(
        shape=grid_shape, dtype=np.int64,
        scratch_dir=scratch_dir, name="grid_shape_nd",
    )
    # Pass 2: slot-assign. Preserves the original semantic of
    # point_ids = np.arange(0, total_points) across the ordered
    # snapshots (bitwise identical to the former np.arange /
    # np.concatenate construction).
    point_offset = 0
    row_offset = 0
    applied_set: set[int] = set()
    reciprocal_point_count = 0
    for partition_id, snapshot_seq, n_points, n_rows in per_snapshot_sizes:
        snapshot = load_local_accumulator_snapshot(
            output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            partition_id=partition_id,
            snapshot_seq=snapshot_seq,
        )
        if snapshot is None:
            raise RuntimeError(
                "Residual-field local finalization lost a snapshot during "
                f"publish: chunk={int(chunk_id)} partition={partition_id} "
                f"seq={int(snapshot_seq)}."
            )
        delta_block = np.asarray(
            snapshot["amplitudes_delta"], dtype=np.complex128
        ).reshape(-1)
        average_block = np.asarray(
            snapshot["amplitudes_average"], dtype=np.complex128
        ).reshape(-1)
        final_delta[point_offset:point_offset + n_points] = delta_block
        final_average[point_offset:point_offset + n_points] = average_block
        expected_point_ids = np.arange(
            point_offset, point_offset + n_points, dtype=np.int64
        )
        snapshot_point_ids = np.asarray(
            snapshot["point_ids"], dtype=np.int64
        ).reshape(-1)
        if (
            snapshot_point_ids.shape == expected_point_ids.shape
            and np.array_equal(snapshot_point_ids, expected_point_ids)
        ):
            final_point_ids[point_offset:point_offset + n_points] = snapshot_point_ids
        else:
            # Partition snapshots use task-local point offsets. The
            # public chunk payload uses contiguous residual row IDs
            # after concatenating partitions.
            final_point_ids[point_offset:point_offset + n_points] = expected_point_ids
        del expected_point_ids, snapshot_point_ids
        grid_block = np.asarray(snapshot["grid_shape_nd"], dtype=np.int64)
        if final_grid_shape_nd.ndim == 2 and grid_block.ndim == 1:
            grid_block = grid_block.reshape(1, -1)
        final_grid_shape_nd[row_offset:row_offset + n_rows] = grid_block
        point_offset += n_points
        row_offset += n_rows
        reciprocal_point_count += int(snapshot["reciprocal_point_count"])
        applied_set.update(
            int(interval_id)
            for interval_id in snapshot["incorporated_interval_ids"]
        )
        del snapshot, delta_block, average_block, grid_block
    snapshot_payload = {
        "point_ids": final_point_ids,
        "grid_shape_nd": final_grid_shape_nd,
        "amplitudes_delta": final_delta,
        "amplitudes_average": final_average,
        "reciprocal_point_count": int(reciprocal_point_count),
        "total_reciprocal_points": int(total_reciprocal_points or 0),
        "incorporated_interval_ids": tuple(sorted(applied_set)),
    }
    return snapshot_payload


def assemble_interval_partitioned_chunk_payload(
    *,
    snapshot_metadata,
    chunk_id,
    parameter_digest,
    output_dir,
    scratch_dir,
    expected_interval_ids=None,
) -> dict | None:
    """Merge a subchunk (interval-partitioned) snapshot family by SUMMATION.

    The transpose of ``assemble_local_snapshot_chunk_payload``: every subchunk
    snapshot covers the chunk's full point range with contributions from a
    disjoint interval subset, so the chunk field is their elementwise sum and
    the incorporated set is their union. Snapshots are loaded one at a time;
    peak finalize RAM is the output buffers plus one snapshot. Family safety
    (disjointness, identical extents) is validated first, and when the plan's
    interval set is provided the union must match it exactly -- a K-of-N
    union is still disjoint, so disjointness alone cannot see a missing
    subchunk's intervals."""
    validate_interval_partitioned_snapshot_family(
        snapshot_metadata,
        chunk_id=chunk_id,
    )
    if expected_interval_ids is not None:
        _require_expected_interval_coverage(
            snapshot_metadata,
            expected_interval_ids=expected_interval_ids,
            chunk_id=chunk_id,
        )
    final_point_ids = None
    final_delta = None
    final_average = None
    final_grid_shape_nd = None
    total_reciprocal_points: int | None = None
    reciprocal_point_count = 0
    applied_set: set[int] = set()
    n_points: int | None = None
    loaded_any = False
    for partition_id, snapshot_seq, _metadata in snapshot_metadata:
        snapshot = load_local_accumulator_snapshot(
            output_dir,
            chunk_id=chunk_id,
            parameter_digest=parameter_digest,
            partition_id=partition_id,
            snapshot_seq=snapshot_seq,
        )
        if snapshot is None:
            # Unlike the concatenating assembler, a listed-but-unreadable
            # snapshot cannot be skipped here: its intervals passed the
            # coverage check on metadata, so dropping its payload would
            # silently publish a chunk missing those contributions.
            raise RuntimeError(
                "Residual-field streaming finalization lost a subchunk "
                f"snapshot during publish: chunk={int(chunk_id)} "
                f"subchunk={partition_id} seq={int(snapshot_seq)}."
            )
        delta_block = np.asarray(
            snapshot["amplitudes_delta"], dtype=np.complex128
        ).reshape(-1)
        average_block = np.asarray(
            snapshot["amplitudes_average"], dtype=np.complex128
        ).reshape(-1)
        if final_delta is None:
            n_points = int(delta_block.shape[0])
            final_delta = _allocate_finalize_output(
                shape=(n_points,), dtype=np.complex128,
                scratch_dir=scratch_dir, name="delta",
            )
            final_average = _allocate_finalize_output(
                shape=(n_points,), dtype=np.complex128,
                scratch_dir=scratch_dir, name="average",
            )
            final_delta[:] = delta_block
            final_average[:] = average_block
            final_point_ids = np.arange(n_points, dtype=np.int64)
            grid_block = np.asarray(snapshot["grid_shape_nd"], dtype=np.int64)
            final_grid_shape_nd = (
                grid_block.reshape(1, -1) if grid_block.ndim == 1 else grid_block
            )
        else:
            if int(delta_block.shape[0]) != n_points:
                raise RuntimeError(
                    "Residual-field streaming finalization found subchunk "
                    f"snapshots of different lengths for chunk={int(chunk_id)} "
                    f"({int(delta_block.shape[0])} vs {n_points}). Stale "
                    "checkpoints from a different layout are present; delete "
                    "'residual_checkpoints/' under the output directory and "
                    "re-run to recompute this chunk."
                )
            final_delta += delta_block
            final_average += average_block
        reciprocal_point_count += int(snapshot["reciprocal_point_count"])
        if total_reciprocal_points is None:
            total_reciprocal_points = int(snapshot["total_reciprocal_points"])
        applied_set.update(
            int(interval_id)
            for interval_id in snapshot["incorporated_interval_ids"]
        )
        loaded_any = True
        del snapshot, delta_block, average_block
    if not loaded_any:
        return None
    return {
        "point_ids": final_point_ids,
        "grid_shape_nd": final_grid_shape_nd,
        "amplitudes_delta": final_delta,
        "amplitudes_average": final_average,
        "reciprocal_point_count": int(reciprocal_point_count),
        "total_reciprocal_points": int(total_reciprocal_points or 0),
        "incorporated_interval_ids": tuple(sorted(applied_set)),
    }
