from .base import DisplayOptions, OverlayOptions, Renderer
from ..widgets.scene_view import SceneView as Renderer2D

_renderer_3d_error: str | None = None

try:
    from .renderer3d import Renderer3D  # type: ignore
    _renderer_3d_available = True
except Exception as exc:  # pragma: no cover - optional dependency
    Renderer3D = None  # type: ignore
    _renderer_3d_available = False
    _renderer_3d_error = str(exc)


def renderer3d_available() -> bool:
    return _renderer_3d_available


def renderer3d_error() -> str | None:
    if _renderer_3d_available:
        return None
    return _renderer_3d_error


__all__ = [
    "DisplayOptions",
    "OverlayOptions",
    "Renderer",
    "Renderer2D",
    "Renderer3D",
    "renderer3d_available",
    "renderer3d_error",
]
