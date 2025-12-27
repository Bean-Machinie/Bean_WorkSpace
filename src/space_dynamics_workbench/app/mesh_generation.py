from __future__ import annotations

from typing import Iterable, List

import numpy as np


def generate_mass_points_from_vertices(vertices: Iterable[Iterable[float]], max_points: int = 5) -> List[np.ndarray]:
    vertices_arr = np.asarray(list(vertices), dtype=float)
    if vertices_arr.size == 0:
        return []
    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("Vertices must be a Nx3 array")
    if max_points <= 0:
        return []

    v_min = vertices_arr.min(axis=0)
    v_max = vertices_arr.max(axis=0)
    center = 0.5 * (v_min + v_max)
    extents = v_max - v_min

    # v1 placeholder: center + extremes along principal axes (deterministic, bbox-based).
    points: List[np.ndarray] = [center]
    if max_points == 1:
        return points

    axes = np.argsort(-extents, kind="mergesort")
    for axis in axes:
        if len(points) >= max_points:
            break
        if extents[axis] <= 0:
            continue
        offset = 0.5 * extents[axis]
        for sign in (1.0, -1.0):
            if len(points) >= max_points:
                break
            point = center.copy()
            point[axis] += sign * offset
            points.append(point)
    return points[:max_points]
