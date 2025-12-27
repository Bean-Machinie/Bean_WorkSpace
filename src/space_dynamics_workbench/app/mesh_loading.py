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
    data = MeshData(
        vertices=vertices,
        faces=faces,
        vertex_count=int(vertices.shape[0]),
        face_count=int(faces.shape[0]),
    )
    _MESH_CACHE[resolved] = data
    return data
