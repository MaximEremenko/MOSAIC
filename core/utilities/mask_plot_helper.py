# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:38:08 2025

@author: Maksim Eremenko
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def plot_mask_3d(hkl_mesh: np.ndarray,
                 mask: np.ndarray,
                 show_false: bool = False,
                 max_points: int = 100_000,
                 true_size: float = 2.0,
                 false_size: float = 1.0):
    """
    Scatter-plot points where mask == True in 3D.

    Args:
        hkl_mesh: (N, 3) array of [h,k,l] points.
        mask: (N,) boolean array.
        show_false: also plot False points faintly (downsampled).
        max_points: cap number of points plotted for each class for speed.
        true_size: marker size for True points.
        false_size: marker size for False points.
    """
    if hkl_mesh.ndim != 2 or hkl_mesh.shape[1] != 3:
        raise ValueError("hkl_mesh must be an (N,3) array")

    # Split
    pts_true = hkl_mesh[mask]
    pts_false = hkl_mesh[~mask] if show_false else None

    rng = np.random.default_rng(0)

    # Downsample for speed if needed
    if len(pts_true) > max_points:
        idx = rng.choice(len(pts_true), size=max_points, replace=False)
        pts_true = pts_true[idx]
    if show_false and len(pts_false) > max_points:
        idx = rng.choice(len(pts_false), size=max_points, replace=False)
        pts_false = pts_false[idx]

    # Figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot True points
    ax.scatter(pts_true[:, 0], pts_true[:, 1], pts_true[:, 2],
               s=true_size, c='tab:blue', alpha=0.95, depthshade=False, label='mask=True')

    # Optionally plot False points (light + transparent)
    if show_false and len(pts_false) > 0:
        ax.scatter(pts_false[:, 0], pts_false[:, 1], pts_false[:, 2],
                   s=false_size, c='lightgray', alpha=0.15, depthshade=False, label='mask=False')

    # Axes labels
    ax.set_xlabel('h')
    ax.set_ylabel('k')
    ax.set_zlabel('l')

    # Equal aspect
    mins = hkl_mesh.min(axis=0)
    maxs = hkl_mesh.max(axis=0)
    centers = (mins + maxs) / 2.0
    half_ranges = (maxs - mins) / 2.0
    r = float(np.max(half_ranges))
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] - r, centers[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))  # mpl >= 3.3
    except Exception:
        pass

    ax.legend(loc='upper left')
    ax.set_title('3D view of mask==True')
    plt.tight_layout()
    plt.show()


# --- Example usage with your EqBasedStrategy output ---
# Suppose you already have: hkl_mesh = (N,3) and mask = strat.generate_mask(hkl_mesh)
# plot_mask_3d(hkl_mesh, mask, show_false=True, max_points=200_000)

def plot_mask_pyvista(hkl_mesh: np.ndarray,
                      mask: np.ndarray,
                      point_size: float = 6.0,
                      axis_labels=("h", "k", "l")):
    """
    Plot True points from a boolean mask using PyVista with axis labels.

    Args:
        hkl_mesh: (N,3) array of [h,k,l] points.
        mask: (N,) boolean array.
        point_size: sphere size for points.
        axis_labels: tuple of strings for the axis labels (h, k, l).
    """
    import pyvista as pv

    pts = hkl_mesh[mask]
    if pts.size == 0:
        raise ValueError("No True points to plot.")

    cloud = pv.PolyData(pts)

    p = pv.Plotter()
    p.add_mesh(cloud, render_points_as_spheres=True, point_size=point_size, opacity=1.0)

    # Axes/grid labels on the bounding box
    # (use show_grid when available; fallback to show_bounds for older PyVista)
    try:
        p.show_grid(xtitle=axis_labels[0], ytitle=axis_labels[1], ztitle=axis_labels[2])
    except TypeError:
        p.show_bounds(xtitle=axis_labels[0], ytitle=axis_labels[1], ztitle=axis_labels[2])

    # Orientation axes widget (bottom-right)
    p.add_axes()

    # Add 3D labels near the positive ends of each axis so "h/k/l" are visible in-scene
    xmin, xmax, ymin, ymax, zmin, zmax = cloud.bounds
    tip_points = np.array([
        [xmax, (ymin + ymax) * 0.5, zmin],  # h
        [(xmin + xmax) * 0.5, ymax, zmin],  # k
        [xmin, ymin, zmax]                  # l
    ])
    p.add_point_labels(
        tip_points,
        list(axis_labels),
        font_size=14,
        point_size=0,
        always_visible=True
    )

    p.show()


def plot_mask_pyvista_to_matplotlib(
    hkl_mesh: np.ndarray,
    mask: np.ndarray,
    point_size: float = 6.0,
    axis_labels=("h", "k", "l"),
    camera: str | tuple = "iso",      # 'iso', 'xy', 'xz', 'yz' or (pos, foc, up)
    figsize=(6, 6),                    # inches
    dpi: int = 150,                    # figure DPI
    axes_off: bool = True,
    transparent_bg: bool = False,
    add_tip_labels: bool = True,
):
    """
    Render the point cloud with PyVista off-screen, then display it in a Matplotlib figure.

    Returns
    -------
    fig, ax, img : (matplotlib.figure.Figure, matplotlib.axes.Axes, np.ndarray)
        img is an RGB(A) numpy array from the PyVista screenshot.
    """
    import pyvista as pv

    pts = hkl_mesh[mask]
    if pts.size == 0:
        raise ValueError("No True points to plot.")

    # Compute pixel window size to match the Matplotlib figure size
    win_w = int(figsize[0] * dpi)
    win_h = int(figsize[1] * dpi)

    # Create off-screen plotter (no interactive window)
    try:
        p = pv.Plotter(off_screen=True, window_size=(win_w, win_h))
    except Exception:
        # Headless Linux fallback
        pv.start_xvfb()
        p = pv.Plotter(off_screen=True, window_size=(win_w, win_h))

    cloud = pv.PolyData(pts)
    p.add_mesh(cloud, render_points_as_spheres=True, point_size=point_size, opacity=1.0)

    # Grid / bounds with axis labels
    try:
        p.show_grid(xtitle=axis_labels[0], ytitle=axis_labels[1], ztitle=axis_labels[2])
    except TypeError:
        p.show_bounds(xtitle=axis_labels[0], ytitle=axis_labels[1], ztitle=axis_labels[2])

    # Small orientation widget
    p.add_axes()

    # Optional in-scene tip labels
    if add_tip_labels:
        xmin, xmax, ymin, ymax, zmin, zmax = cloud.bounds
        tip_points = np.array([
            [xmax, 0.5*(ymin+ymax), zmin],  # h
            [0.5*(xmin+xmax), ymax, zmin],  # k
            [xmin, ymin, zmax]              # l
        ])
        p.add_point_labels(
            tip_points,
            list(axis_labels),
            font_size=14,
            point_size=0,
            always_visible=True
        )

    # Camera view
    if isinstance(camera, str):
        p.camera_position = camera
    else:
        p.camera_position = camera  # (pos, foc, up)

    # Take a screenshot as a numpy array
    img = p.screenshot(transparent_background=transparent_bg, return_img=True)
    p.close()

    # Show in a Matplotlib window
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img)
    if axes_off:
        ax.axis("off")
    else:
        ax.set_xlabel("pixels (x)")
        ax.set_ylabel("pixels (y)")
    fig.tight_layout()
    plt.show()

    return fig, ax, img