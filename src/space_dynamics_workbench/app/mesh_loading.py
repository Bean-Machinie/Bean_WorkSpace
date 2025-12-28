from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_TRIMESH_ERROR: Optional[str] = None
try:  # pragma: no cover - optional dependency
    import trimesh  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    trimesh = None  # type: ignore
    _TRIMESH_ERROR = str(exc)


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None
    vertex_count: int
    face_count: int


def mesh_loading_available() -> bool:
    return trimesh is not None


def mesh_loading_error() -> str | None:
    if trimesh is None:
        return _TRIMESH_ERROR or "trimesh is not installed"
    return None


_MESH_CACHE: dict[str, MeshData] = {}


def load_mesh_data(path: Path) -> MeshData:
    if trimesh is None:
        raise RuntimeError(mesh_loading_error() or "Mesh loading is unavailable")
    resolved = str(path.resolve())
    cached = _MESH_CACHE.get(resolved)
    if cached is not None:
        return cached

    mesh = trimesh.load(resolved, force="mesh", process=False)
    if hasattr(mesh, "geometry"):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise ValueError(f"No geometry found in mesh: {resolved}")
        mesh = trimesh.util.concatenate(geometries)

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertex_colors = _extract_vertex_colors(mesh, vertices.shape[0])
    data = MeshData(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        vertex_count=int(vertices.shape[0]),
        face_count=int(faces.shape[0]),
    )
    _MESH_CACHE[resolved] = data
    return data


def _extract_vertex_colors(mesh: object, vertex_count: int) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    colors = getattr(visual, "vertex_colors", None)
    if colors is None or len(colors) == 0:
        return None
    colors_arr = np.asarray(colors)
    if colors_arr.shape[0] != vertex_count:
        return None
    if colors_arr.ndim != 2 or colors_arr.shape[1] not in (3, 4):
        return None
    return colors_arr
