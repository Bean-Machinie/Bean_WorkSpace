from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore, QtWidgets


@dataclass
class OverlayOptions:
    show_r_op: bool = True
    show_r_oc: bool = True
    show_r_cp: bool = False
    show_grid_xy: bool = True
    show_grid_xz: bool = False
    show_grid_yz: bool = False


@dataclass
class DisplayOptions:
    show_mass_points: bool = True
    show_mesh: bool = True
    mesh_opacity: float = 0.35


class Renderer(QtWidgets.QWidget):
    entity_selected = QtCore.Signal(str)
    entity_dragged = QtCore.Signal(str, object)

    def set_scene(
        self,
        frame_vectors,
        overlays: OverlayOptions,
        display: DisplayOptions | None,
        selected_id: str | None,
        selected_component_id: str | None = None,
        entities=None,
    ) -> None:
        raise NotImplementedError

    def frame_scene(self) -> None:
        pass

    def set_view_range(self, view_range: tuple[float, float, float, float]) -> None:
        _ = view_range

    def clear(self) -> None:
        raise NotImplementedError
