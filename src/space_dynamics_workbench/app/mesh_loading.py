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

    loaded = trimesh.load(resolved, force="scene", process=True)
    if hasattr(loaded, "to_geometry"):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertex_colors = _colors_for_visual(getattr(mesh, "visual", None), vertices.shape[0])
    data = MeshData(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors,
        vertex_count=int(vertices.shape[0]),
        face_count=int(faces.shape[0]),
    )
    _MESH_CACHE[resolved] = data
    return data


def _colors_for_visual(visual: object | None, vertex_count: int) -> np.ndarray | None:
    if visual is None:
        return None
    colors = _colors_from_visual_colors(visual, vertex_count)
    if colors is not None:
        return colors
    colors = _colors_from_texture(visual, vertex_count)
    if colors is not None:
        return colors
    material = getattr(visual, "material", None)
    base = getattr(material, "baseColorFactor", None) if material is not None else None
    if base is None:
        return None
    base_arr = np.asarray(base, dtype=float).reshape(-1)
    if base_arr.size not in (3, 4):
        return None
    if base_arr.max() <= 1.0:
        base_arr = base_arr * 255.0
    if base_arr.size == 3:
        base_arr = np.array([base_arr[0], base_arr[1], base_arr[2], 255.0], dtype=float)
    base_arr = np.clip(base_arr, 0.0, 255.0).astype(np.uint8)
    return np.tile(base_arr, (vertex_count, 1))


def _colors_from_visual_colors(visual: object, vertex_count: int) -> np.ndarray | None:
    colors = getattr(visual, "vertex_colors", None)
    if colors is None or len(colors) == 0:
        return None
    colors_arr = np.asarray(colors)
    if colors_arr.shape[0] != vertex_count:
        return None
    if colors_arr.ndim != 2 or colors_arr.shape[1] not in (3, 4):
        return None
    return colors_arr


def _colors_from_texture(visual: object, vertex_count: int) -> np.ndarray | None:
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = _image_from_material(material)
    if uv is None or image is None:
        return None
    uv_arr = np.asarray(uv, dtype=float)
    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2 or uv_arr.shape[0] != vertex_count:
        return None
    image_arr = np.asarray(image)
    if image_arr.ndim == 2:
        image_arr = np.stack([image_arr] * 3, axis=-1)
    if image_arr.ndim != 3 or image_arr.shape[2] not in (3, 4):
        return None
    height, width = image_arr.shape[0], image_arr.shape[1]
    u = np.clip(uv_arr[:, 0], 0.0, 1.0)
    v = np.clip(uv_arr[:, 1], 0.0, 1.0)
    x = np.clip((u * (width - 1)).round().astype(int), 0, width - 1)
    y = np.clip(((1.0 - v) * (height - 1)).round().astype(int), 0, height - 1)
    return image_arr[y, x]


def _image_from_material(material: object | None) -> object | None:
    if material is None:
        return None
    image = getattr(material, "image", None)
    if image is None:
        image = getattr(material, "baseColorTexture", None)
    if image is None and hasattr(material, "baseColorTexture"):
        base_texture = getattr(material, "baseColorTexture", None)
        image = getattr(base_texture, "image", None) if base_texture is not None else None
    return image
