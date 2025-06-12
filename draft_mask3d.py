import numpy as np, math, warnings
from numba import njit, prange, cuda


# ───────────────────────── 1. periodic expansion (CPU, parallel) ────────
@njit(parallel=True)
def _expand_centres3d(centres, radii, cmin, cmax):
    """
    centres : (M,3)  radii : (M,)
    Return all periodic images that can overlap the simulation box.
      out_xyz : (K,3) float64   out_r : (K,) float64
    K ≤ 27 × M  (3 images per axis)
    """
    M = radii.size

    # pass 1 – count images per sphere
    counts = np.empty(M, np.int64)
    for m in range(M):
        xs = find_val_in_interval(cmin[0], cmax[0], centres[m, 0])
        ys = find_val_in_interval(cmin[1], cmax[1], centres[m, 1])
        zs = find_val_in_interval(cmin[2], cmax[2], centres[m, 2])
        counts[m] = xs.size * ys.size * zs.size    # ≤ 27

    K = counts.sum()

    # prefix sum → offsets
    offsets = np.empty(M + 1, np.int64)
    offsets[0] = 0
    for m in range(M):
        offsets[m + 1] = offsets[m] + counts[m]

    out_xyz = np.empty((K, 3), np.float64)
    out_r   = np.empty(K,       np.float64)

    # pass 2 – fill (parallel)
    for m in prange(M):
        xs = find_val_in_interval(cmin[0], cmax[0], centres[m, 0])
        ys = find_val_in_interval(cmin[1], cmax[1], centres[m, 1])
        zs = find_val_in_interval(cmin[2], cmax[2], centres[m, 2])

        idx = offsets[m]
        for i in range(xs.size):
            for j in range(ys.size):
                for k in range(zs.size):
                    out_xyz[idx, 0] = xs[i]
                    out_xyz[idx, 1] = ys[j]
                    out_xyz[idx, 2] = zs[k]
                    out_r[idx]      = radii[m]
                    idx += 1

    return out_xyz, out_r


# ───────────────────────── 2. CUDA kernel (thread ↦ point) ──────────────
@cuda.jit
def _mask_kernel3d(points, centres, radii, out_mask):
    """
    points  : (N,3)  centres : (K,3)  radii : (K,)  out_mask : (N,)
    """
    p = cuda.grid(1)
    if p >= points.shape[0]:
        return

    x, y, z = points[p, 0], points[p, 1], points[p, 2]
    hit = 0
    for c in range(centres.shape[0]):
        dx = x - centres[c, 0]
        dy = y - centres[c, 1]
        dz = z - centres[c, 2]
        if dx*dx + dy*dy + dz*dz <= radii[c]*radii[c]:
            hit = 1
            break
    out_mask[p] = hit


# ───────────────────────── 3. public GPU wrapper  ───────────────────────
def compute_3d_mask_gpu(data_points: np.ndarray,   # (N,3)
                        radii:       np.ndarray,   # (M,)
                        centres:     np.ndarray,   # (M,3)
                        coord_min:   np.ndarray,   # (3,)
                        coord_max:   np.ndarray    # (3,)
                        ) -> np.ndarray:
    """
    GPU-accelerated Boolean mask for spheres with periodic images.
    """
    # build periodic images (CPU, fast)
    exp_centres, exp_r = _expand_centres3d(centres, radii,
                                           coord_min, coord_max)

    # move to device
    d_pts  = cuda.to_device(data_points.astype(np.float64))
    d_ctr  = cuda.to_device(exp_centres)
    d_rad  = cuda.to_device(exp_r)
    d_mask = cuda.device_array(data_points.shape[0], dtype=np.uint8)

    # launch kernel – 1 thread per point
    TPB    = 128
    blocks = math.ceil(data_points.shape[0] / TPB)
    _mask_kernel3d[blocks, TPB](d_pts, d_ctr, d_rad, d_mask)

    return d_mask.copy_to_host().view(np.bool_)


# ───────────────────────── 4. optional CPU fallback  ────────────────────
@njit(parallel=True, fastmath=True)
def compute_3d_mask_cpu(data_points, radii, centres, cmin, cmax):
    """
    Fully threaded CPU version (similar logic, slower than GPU).
    """
    N, M = data_points.shape[0], radii.size
    mask = np.zeros(N, np.bool_)

    for m in prange(M):
        r2 = radii[m]*radii[m]
        xs = find_val_in_interval(cmin[0], cmax[0], centres[m, 0])
        ys = find_val_in_interval(cmin[1], cmax[1], centres[m, 1])
        zs = find_val_in_interval(cmin[2], cmax[2], centres[m, 2])
        for p in range(N):                # each thread iterates its slice
            hit = False
            for i in range(xs.size):
                dx = data_points[p, 0] - xs[i]
                dx2 = dx*dx
                for j in range(ys.size):
                    dy = data_points[p, 1] - ys[j]
                    for k in range(zs.size):
                        dz = data_points[p, 2] - zs[k]
                        if dx2 + dy*dy + dz*dz <= r2:
                            hit = True
                            break
                    if hit: break
                if hit: break
            if hit:
                mask[p] = True
    return mask


# ───────────────────────── 5. dispatcher (GPU first, else CPU) ──────────
def compute_3d_mask(data_points, radii, centres,
                    coord_min, coord_max,
                    gpu_threshold=20_000):
    """
    Call this everywhere; it picks the fastest backend automatically.
    """
    if cuda.is_available() and data_points.shape[0] >= gpu_threshold:
        try:
            return compute_3d_mask_gpu(data_points, radii, centres,
                                       coord_min, coord_max)
        except cuda.cudadrv.error.CudaSupportError as e:
            warnings.warn(f"CUDA unavailable ({e}); falling back to CPU")

    return compute_3d_mask_cpu(data_points, radii, centres,
                               coord_min, coord_max)
