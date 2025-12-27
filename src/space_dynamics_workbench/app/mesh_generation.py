from __future__ import annotations

from typing import Iterable, List, Sequence

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


def generate_mass_points_from_mesh(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]] | None,
    max_points: int = 5,
) -> List[np.ndarray]:
    vertices_arr = np.asarray(vertices, dtype=float)
    if vertices_arr.size == 0:
        return []
    if faces is None or len(faces) == 0:
        return generate_mass_points_from_vertices(vertices_arr, max_points=max_points)
    faces_arr = np.asarray(faces, dtype=int)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("Faces must be a Nx3 array")

    areas = _triangle_areas(vertices_arr, faces_arr)
    if areas.size == 0:
        return generate_mass_points_from_vertices(vertices_arr, max_points=max_points)
    top_k = int(min(len(faces_arr), max(20, max_points * 10)))
    top_indices = np.argsort(-areas, kind="mergesort")[:top_k]
    candidates = vertices_arr[faces_arr[top_indices]].mean(axis=1)
    if candidates.size == 0:
        return generate_mass_points_from_vertices(vertices_arr, max_points=max_points)
    points = _farthest_point_subset(candidates, max_points=max_points)
    return points


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    vec1 = tri[:, 1] - tri[:, 0]
    vec2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(vec1, vec2)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _farthest_point_subset(points: np.ndarray, max_points: int) -> List[np.ndarray]:
    if points.size == 0 or max_points <= 0:
        return []
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    first_idx = int(np.argmax(distances))
    selected = [points[first_idx]]
    if max_points == 1:
        return selected
    remaining = points.copy()
    remaining = np.delete(remaining, first_idx, axis=0)
    while remaining.size > 0 and len(selected) < max_points:
        dist_to_selected = np.min(
            np.linalg.norm(remaining[:, None, :] - np.asarray(selected)[None, :, :], axis=2),
            axis=1,
        )
        next_idx = int(np.argmax(dist_to_selected))
        selected.append(remaining[next_idx])
        remaining = np.delete(remaining, next_idx, axis=0)
    return selected
