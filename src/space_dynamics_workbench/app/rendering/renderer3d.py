from __future__ import annotations

from pathlib import Path
from typing import List

import logging
import os

import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets

from ...core.model import MeshMetadata, PointMass, RigidBody, resolve_entity_by_id
from ...core.physics import FrameVectors, VectorSegment
from ..mesh_loading import load_mesh_data, mesh_loading_available
from .base import DisplayOptions, OverlayOptions, Renderer


class _PickingGLViewWidget(gl.GLViewWidget):
    def __init__(self, on_click, on_press=None, on_move=None, on_release=None, parent=None) -> None:
        super().__init__(parent=parent)
        self._on_click = on_click
        self._on_press = on_press
        self._on_move = on_move
        self._on_release = on_release
        self._press_pos: QtCore.QPointF | None = None
        self._press_handled = False

    def mousePressEvent(self, ev) -> None:
        self._press_pos = ev.position() if hasattr(ev, "position") else ev.localPos()
        handled = False
        if self._on_press is not None:
            handled = bool(self._on_press(ev))
        self._press_handled = handled
        if handled:
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev) -> None:
        if self._on_move is not None:
            handled = bool(self._on_move(ev))
            if handled:
                ev.accept()
                return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        if self._press_handled:
            if self._on_release is not None:
                self._on_release(ev)
            self._press_handled = False
            self._press_pos = None
            ev.accept()
            return
        release_pos = ev.position() if hasattr(ev, "position") else ev.localPos()
        if self._press_pos is not None and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if (release_pos - self._press_pos).manhattanLength() < 4.0:
                handled = bool(self._on_click(release_pos))
                if handled:
                    self._press_pos = None
                    ev.accept()
                    return
        self._press_pos = None
        super().mouseReleaseEvent(ev)


class _OrientationWidget(QtWidgets.QFrame):
    def __init__(self, on_select, on_orbit, parent=None) -> None:
        super().__init__(parent=parent)
        self._on_select = on_select
        self._on_orbit = on_orbit
        self._dragging = False
        self._last_pos: QtCore.QPoint | None = None
        self.setObjectName("orientationWidget")
        self.setStyleSheet(
            "#orientationWidget {"
            " background-color: rgba(24, 24, 24, 210);"
            " border: 1px solid rgba(255, 255, 255, 40);"
            " border-radius: 8px;"
            "}"
            "#orientationWidget QToolButton {"
            " min-width: 28px;"
            " min-height: 22px;"
            " padding: 2px 4px;"
            " border: 1px solid rgba(255, 255, 255, 35);"
            " border-radius: 4px;"
            " color: #e6e6e6;"
            " background-color: rgba(40, 40, 40, 200);"
            "}"
            "#orientationWidget QToolButton:hover {"
            " background-color: rgba(65, 65, 65, 230);"
            "}"
            "#orientationWidget QToolButton#axisX { color: #ff5b5b; }"
            "#orientationWidget QToolButton#axisY { color: #54d46a; }"
            "#orientationWidget QToolButton#axisZ { color: #5aa9ff; }"
        )

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        buttons = [
            ("-X", "axisX", "-x"),
            ("X", "axisX", "x"),
            ("-Y", "axisY", "-y"),
            ("Y", "axisY", "y"),
            ("-Z", "axisZ", "-z"),
            ("Z", "axisZ", "z"),
        ]
        for idx, (label, obj_name, key) in enumerate(buttons):
            button = QtWidgets.QToolButton(self)
            button.setText(label)
            button.setObjectName(obj_name)
            button.clicked.connect(lambda _checked=False, k=key: self._on_select(k))
            layout.addWidget(button, idx // 2, idx % 2)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._can_orbit(event.pos()):
            self._dragging = True
            self._last_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self._on_orbit(delta.x(), delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = False
            self._last_pos = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _can_orbit(self, pos: QtCore.QPoint) -> bool:
        child = self.childAt(pos)
        return child is None


class Renderer3D(Renderer):
    transform_requested = QtCore.Signal(str, str, object)
    GRID_SIZE = 60.0
    GRID_SPACING_MINOR = 1.0
    GRID_SPACING_MAJOR = 5.0
    FRAME_DISTANCE_MULTIPLIER = 2.8
    FRAME_MIN_DISTANCE = 6.0
    PICK_RADIUS_PX = 12.0
    BG_COLOR = "#1f1f1f"
    GRID_MINOR_COLOR = (200, 200, 200, 70)
    GRID_MAJOR_COLOR = (220, 220, 220, 120)
    GRID_SIDE_COLOR = (200, 200, 200, 50)
    POINT_COLOR = (0.45, 0.72, 0.98, 1.0)
    POINT_SELECTED_COLOR = (1.0, 0.62, 0.2, 1.0)
    POINT_COMPONENT_COLOR = (0.3, 0.85, 0.4, 1.0)
    ORIGIN_COLOR = (0.55, 0.55, 0.55, 1.0)
    COM_COLOR = (1.0, 0.3, 0.3, 1.0)
    MESH_BASE_COLOR = (0.75, 0.78, 0.82, 1.0)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = logging.getLogger(__name__)
        self._view = _PickingGLViewWidget(
            self._handle_click,
            on_press=self._handle_mouse_press,
            on_move=self._handle_mouse_move,
            on_release=self._handle_mouse_release,
        )
        self._view.setBackgroundColor(self.BG_COLOR)
        self._view.opts["distance"] = 20.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._axis_item = gl.GLAxisItem(size=QtGui.QVector3D(10.0, 10.0, 10.0))
        self._view.addItem(self._axis_item)
        self._axis_lines = self._make_axis_lines()
        for axis_line in self._axis_lines:
            self._view.addItem(axis_line)
        self._axis_labels = self._make_axis_labels()
        for label in self._axis_labels:
            self._view.addItem(label)

        self._orientation_widget = _OrientationWidget(self.set_orientation, self._orbit_from_gizmo, self._view)
        self._orientation_margin = 12
        self._position_overlay()

        self._grid_xy_minor = self._make_grid(self.GRID_SPACING_MINOR, self.GRID_MINOR_COLOR)
        self._grid_xy_major = self._make_grid(self.GRID_SPACING_MAJOR, self.GRID_MAJOR_COLOR)
        self._grid_xz_minor = self._make_grid(self.GRID_SPACING_MINOR, self.GRID_SIDE_COLOR)
        self._grid_xz_major = self._make_grid(self.GRID_SPACING_MAJOR, self.GRID_SIDE_COLOR)
        self._grid_yz_minor = self._make_grid(self.GRID_SPACING_MINOR, self.GRID_SIDE_COLOR)
        self._grid_yz_major = self._make_grid(self.GRID_SPACING_MAJOR, self.GRID_SIDE_COLOR)

        self._grid_xz_minor.rotate(90, 1, 0, 0)
        self._grid_xz_major.rotate(90, 1, 0, 0)
        self._grid_yz_minor.rotate(90, 0, 1, 0)
        self._grid_yz_major.rotate(90, 0, 1, 0)

        for grid in (
            self._grid_xy_minor,
            self._grid_xy_major,
            self._grid_xz_minor,
            self._grid_xz_major,
            self._grid_yz_minor,
            self._grid_yz_major,
        ):
            self._view.addItem(grid)

        self._points_item = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3)),
            size=7.0,
            pxMode=True,
            glOptions="opaque",
        )
        self._origin_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=self.ORIGIN_COLOR,
            size=7.0,
            pxMode=True,
            glOptions="opaque",
        )
        self._com_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=self.COM_COLOR,
            size=11.0,
            pxMode=True,
            glOptions="opaque",
        )
        self._rb_com_item = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3)),
            color=(1.0, 0.55, 0.2, 0.9),
            size=9.0,
            pxMode=True,
            glOptions="opaque",
        )

        self._r_op_item = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)),
            color=(0.1, 0.46, 0.82, 1.0),
            width=1.2,
            antialias=True,
            mode="lines",
            glOptions="opaque",
        )
        self._r_cp_item = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)),
            color=(0.22, 0.56, 0.24, 1.0),
            width=1.2,
            antialias=True,
            mode="lines",
            glOptions="opaque",
        )
        self._r_oc_item = gl.GLLinePlotItem(
            pos=np.zeros((0, 3)),
            color=(0.83, 0.18, 0.18, 1.0),
            width=1.4,
            antialias=True,
            mode="lines",
            glOptions="opaque",
        )

        self._view.addItem(self._points_item)
        self._view.addItem(self._origin_item)
        self._view.addItem(self._com_item)
        self._view.addItem(self._rb_com_item)
        self._view.addItem(self._r_op_item)
        self._view.addItem(self._r_cp_item)
        self._view.addItem(self._r_oc_item)

        self._entity_ids: List[str] = []
        self._positions: np.ndarray = np.zeros((0, 3), dtype=float)
        self._last_com: np.ndarray = np.zeros(3, dtype=float)
        self._viewport_size: tuple[int, int] = (1, 1)
        self._rb_com_ids: List[str] = []
        self._rb_com_positions: np.ndarray = np.zeros((0, 3), dtype=float)
        self._mesh_items: dict[str, gl.GLMeshItem] = {}
        self._mesh_sources: dict[str, str] = {}
        self._project_root = Path.cwd()
        self._orientation_timer = QtCore.QTimer(self)
        self._orientation_timer.setInterval(16)
        self._orientation_timer.timeout.connect(self._update_orientation_animation)
        self._orientation_anim = None
        self._interaction_mode = "select"
        self._selected_entity_id: str | None = None
        self._selected_component_id: str | None = None
        self._entities = []
        self._gizmo_pos = np.zeros(3, dtype=float)
        self._gizmo_axis_len = 2.8
        self._gizmo_ring_radius = 2.2
        self._gizmo_target_id: str | None = None
        self._dragging = False
        self._drag_axis: str | None = None
        self._drag_target_id: str | None = None
        self._drag_last_pos: QtCore.QPointF | None = None
        self._drag_axis_dir = np.zeros(3, dtype=float)
        self._drag_screen_dir = np.zeros(2, dtype=float)
        self._drag_scale = 0.01
        self._rotate_scale_deg = 0.35
        self._drag_center_screen: QtCore.QPointF | None = None
        self._drag_prev_vec = np.zeros(2, dtype=float)

        self._gizmo_x = self._make_gizmo_line((1.0, 0.2, 0.2, 1.0))
        self._gizmo_y = self._make_gizmo_line((0.2, 0.85, 0.2, 1.0))
        self._gizmo_z = self._make_gizmo_line((0.2, 0.5, 0.95, 1.0))
        self._gizmo_rx = self._make_gizmo_ring((1.0, 0.45, 0.45, 0.9))
        self._gizmo_ry = self._make_gizmo_ring((0.45, 1.0, 0.55, 0.9))
        self._gizmo_rz = self._make_gizmo_ring((0.45, 0.65, 1.0, 0.9))
        for item in (
            self._gizmo_x,
            self._gizmo_y,
            self._gizmo_z,
            self._gizmo_rx,
            self._gizmo_ry,
            self._gizmo_rz,
        ):
            self._view.addItem(item)

    def set_view_range(self, view_range: tuple[float, float, float, float]) -> None:
        _ = view_range

    def capture_frame(self) -> QtGui.QImage:
        image = self._view.grabFramebuffer()
        if image.isNull():
            return super().capture_frame()
        return image

    def set_scene(
        self,
        frame_vectors: FrameVectors,
        overlays: OverlayOptions,
        display: DisplayOptions | None,
        selected_id: str | None,
        selected_component_id: str | None = None,
        entities=None,
    ) -> None:
        display_options = display or DisplayOptions()
        self._selected_entity_id = selected_id
        self._selected_component_id = selected_component_id
        self._entities = list(entities) if entities is not None else []
        positions = self._to_3d_stack(frame_vectors.positions)
        self._positions = positions
        self._entity_ids = list(frame_vectors.entity_ids)
        self._last_com = self._to_3d(frame_vectors.com)
        self._viewport_size = (max(self._view.width(), 1), max(self._view.height(), 1))

        if self._debug_enabled():
            sample = positions[:3] if positions.size else positions
            self._log.info(
                "Renderer3D set_scene entities=%d sample=%s center=%s distance=%.3f",
                positions.shape[0],
                sample,
                self._view.opts.get("center"),
                float(self._view.opts.get("distance", 0.0)),
            )

        if positions.size == 0:
            self._points_item.setData(pos=np.zeros((0, 3), dtype=np.float32), size=7.0)
        else:
            masses = np.asarray(frame_vectors.masses, dtype=float)
            sizes = 7.0 + 3.0 * np.sqrt(np.maximum(masses, 0.0))
            colors = np.tile(np.array(self.POINT_COLOR, dtype=np.float32), (positions.shape[0], 1))
            if selected_id in self._entity_ids:
                idx = self._entity_ids.index(selected_id)
                colors[idx] = np.array(self.POINT_SELECTED_COLOR, dtype=np.float32)
                sizes[idx] += 2.0
            if selected_component_id in self._entity_ids:
                idx = self._entity_ids.index(selected_component_id)
                colors[idx] = np.array(self.POINT_COMPONENT_COLOR, dtype=np.float32)
                sizes[idx] += 4.0
            self._points_item.setData(pos=positions.astype(np.float32, copy=False), size=sizes, color=colors)
        self._points_item.setVisible(display_options.show_mass_points)

        origin_pos = self._to_3d(frame_vectors.origin)
        self._origin_item.setData(pos=origin_pos.reshape(1, 3))
        self._com_item.setData(pos=self._last_com.reshape(1, 3))
        self._update_rigid_body_coms()

        self._set_vector_item(self._r_op_item, frame_vectors.r_op_segments, overlays.show_r_op)
        self._set_vector_item(self._r_cp_item, frame_vectors.r_cp_segments, overlays.show_r_cp)
        self._set_vector_item(self._r_oc_item, [frame_vectors.r_oc_segment], overlays.show_r_oc)

        self._grid_xy_minor.setVisible(overlays.show_grid_xy)
        self._grid_xy_major.setVisible(overlays.show_grid_xy)
        self._grid_xz_minor.setVisible(overlays.show_grid_xz)
        self._grid_xz_major.setVisible(overlays.show_grid_xz)
        self._grid_yz_minor.setVisible(overlays.show_grid_yz)
        self._grid_yz_major.setVisible(overlays.show_grid_yz)

        axis_visible = overlays.show_axes
        labels_visible = overlays.show_axis_labels and bool(self._axis_labels)
        self._axis_item.setVisible(axis_visible)
        for axis_line in self._axis_lines:
            axis_line.setVisible(axis_visible)
        for label in self._axis_labels:
            label.setVisible(labels_visible and axis_visible)
        self._orientation_widget.setVisible(True)

        self._update_meshes(entities, display_options)
        self._update_gizmo()

    def clear(self) -> None:
        self._points_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._origin_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._com_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._r_op_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._r_cp_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._r_oc_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._last_com = np.zeros(3, dtype=float)
        for mesh_item in self._mesh_items.values():
            self._view.removeItem(mesh_item)
        self._mesh_items.clear()
        self._mesh_sources.clear()

    def frame_scene(self) -> None:
        if self._positions.size == 0 and self._last_com.size == 0:
            return
        center, radius = self._compute_frame_bounds()
        self._auto_fit_camera(center, radius)

    def _set_vector_item(self, item: gl.GLLinePlotItem, segments: List[VectorSegment], visible: bool) -> None:
        item.setVisible(visible)
        if not visible or not segments:
            item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return
        pos = []
        for segment in segments:
            start = self._to_3d(segment.start)
            end = self._to_3d(segment.end)
            pos.append(start)
            pos.append(end)
        item.setData(pos=np.asarray(pos, dtype=np.float32))

    def _to_3d_stack(self, vectors: List[np.ndarray]) -> np.ndarray:
        if not vectors:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([self._to_3d(vec) for vec in vectors]).astype(np.float32, copy=False)

    def _to_3d(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=float).reshape(-1)
        if vec.size >= 3:
            return vec[:3]
        padded = np.zeros(3, dtype=float)
        padded[: vec.size] = vec
        return padded

    def _auto_fit_camera(self, center_vec: np.ndarray, radius: float) -> None:
        distance = max(self.FRAME_MIN_DISTANCE, radius * self.FRAME_DISTANCE_MULTIPLIER)
        center = QtGui.QVector3D(float(center_vec[0]), float(center_vec[1]), float(center_vec[2]))
        self._view.setCameraPosition(pos=center, distance=distance)

    def _compute_frame_bounds(self) -> tuple[np.ndarray, float]:
        if self._positions.size == 0:
            return self._last_com, 1.0
        points = [self._positions]
        if self._last_com.size == 3:
            points.append(self._last_com.reshape(1, 3))
        stacked = np.vstack(points)
        center = self._last_com if self._last_com.size == 3 else stacked.mean(axis=0)
        distances = np.linalg.norm(stacked - center.reshape(1, 3), axis=1)
        radius = float(np.max(distances)) if distances.size else 1.0
        return center, max(radius, 1.0)

    def _make_grid(self, spacing: float, color: tuple[float, float, float, float]) -> gl.GLGridItem:
        grid = gl.GLGridItem()
        grid.setSize(self.GRID_SIZE, self.GRID_SIZE)
        grid.setSpacing(spacing, spacing)
        grid.setColor(color)
        return grid

    def _make_axis_lines(self) -> List[gl.GLLinePlotItem]:
        axis_len = 10.0
        line_data = [
            (np.array([[0.0, 0.0, 0.0], [axis_len, 0.0, 0.0]]), (0.9, 0.2, 0.2, 1.0)),
            (np.array([[0.0, 0.0, 0.0], [0.0, axis_len, 0.0]]), (0.2, 0.85, 0.2, 1.0)),
            (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, axis_len]]), (0.2, 0.5, 0.95, 1.0)),
        ]
        items: List[gl.GLLinePlotItem] = []
        for pos, color in line_data:
            items.append(
                gl.GLLinePlotItem(
                    pos=pos.astype(np.float32),
                    color=color,
                    width=3.0,
                    antialias=True,
                    mode="lines",
                    glOptions="opaque",
                )
            )
        return items

    def _make_axis_labels(self) -> List[gl.GLTextItem]:
        if not hasattr(gl, "GLTextItem"):
            return []
        axis_len = 10.0
        labels = [
            ("X", (axis_len + 0.6, 0.0, 0.0), QtGui.QColor(235, 90, 90)),
            ("Y", (0.0, axis_len + 0.6, 0.0), QtGui.QColor(80, 210, 110)),
            ("Z", (0.0, 0.0, axis_len + 0.6), QtGui.QColor(90, 160, 255)),
        ]
        items: List[gl.GLTextItem] = []
        for text, pos, color in labels:
            items.append(gl.GLTextItem(pos=pos, text=text, color=color))
        return items

    def set_interaction_mode(self, mode: str) -> None:
        if mode not in {"select", "move", "rotate"}:
            return
        self._interaction_mode = mode
        self._update_gizmo()

    def _make_gizmo_line(self, color: tuple[float, float, float, float]) -> gl.GLLinePlotItem:
        return gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=color,
            width=4.0,
            antialias=True,
            mode="lines",
            glOptions="opaque",
        )

    def _make_gizmo_ring(self, color: tuple[float, float, float, float]) -> gl.GLLinePlotItem:
        return gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=color,
            width=2.0,
            antialias=True,
            mode="line_strip",
            glOptions="opaque",
        )

    def _update_gizmo(self) -> None:
        target_id = self._resolve_gizmo_target()
        self._gizmo_target_id = target_id
        if target_id is None or self._interaction_mode == "select":
            self._set_gizmo_visibility(False, False)
            self._rb_com_item.setVisible(False)
            return
        center = self._target_world_position(target_id)
        if center is None:
            self._set_gizmo_visibility(False, False)
            self._rb_com_item.setVisible(False)
            return
        self._gizmo_pos = center
        axis_len = self._gizmo_axis_len
        center_vec = center.astype(np.float32, copy=False)
        if self._interaction_mode == "move":
            self._gizmo_x.setData(pos=self._axis_line_with_arrow(center_vec, np.array([1.0, 0.0, 0.0])))
            self._gizmo_y.setData(pos=self._axis_line_with_arrow(center_vec, np.array([0.0, 1.0, 0.0])))
            self._gizmo_z.setData(pos=self._axis_line_with_arrow(center_vec, np.array([0.0, 0.0, 1.0])))
            self._set_gizmo_visibility(True, False)
            self._rb_com_item.setVisible(True)
            return
        if self._interaction_mode == "rotate":
            ring = self._gizmo_ring_radius
            points = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=True)
            cos_vals = np.cos(points) * ring
            sin_vals = np.sin(points) * ring
            ring_x = np.column_stack((np.zeros_like(cos_vals), cos_vals, sin_vals)).astype(np.float32)
            ring_y = np.column_stack((cos_vals, np.zeros_like(cos_vals), sin_vals)).astype(np.float32)
            ring_z = np.column_stack((cos_vals, sin_vals, np.zeros_like(cos_vals))).astype(np.float32)
            self._gizmo_rx.setData(pos=(ring_x + center_vec))
            self._gizmo_ry.setData(pos=(ring_y + center_vec))
            self._gizmo_rz.setData(pos=(ring_z + center_vec))
            self._set_gizmo_visibility(False, True)
            self._rb_com_item.setVisible(False)

    def _set_gizmo_visibility(self, axes: bool, rings: bool) -> None:
        self._gizmo_x.setVisible(axes)
        self._gizmo_y.setVisible(axes)
        self._gizmo_z.setVisible(axes)
        self._gizmo_rx.setVisible(rings)
        self._gizmo_ry.setVisible(rings)
        self._gizmo_rz.setVisible(rings)

    def _axis_line_with_arrow(self, center: np.ndarray, axis_dir: np.ndarray) -> np.ndarray:
        axis = np.asarray(axis_dir, dtype=float)
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        end = center + axis * self._gizmo_axis_len
        head_len = 0.45
        head_width = 0.22
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(axis, up))) > 0.9:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        side = np.cross(axis, up)
        side = side / max(np.linalg.norm(side), 1e-9)
        left = end - axis * head_len + side * head_width
        right = end - axis * head_len - side * head_width
        points = np.vstack([center, end, end, left, end, right]).astype(np.float32)
        return points

    def _resolve_gizmo_target(self) -> str | None:
        if self._selected_entity_id is None:
            return None
        if self._interaction_mode == "move" and self._selected_component_id is not None:
            return self._selected_component_id
        if self._interaction_mode == "rotate":
            entity, _ = resolve_entity_by_id(self._entities, self._selected_entity_id)
            if isinstance(entity, RigidBody):
                return entity.entity_id
            return None
        return self._selected_entity_id

    def _target_world_position(self, target_id: str) -> np.ndarray | None:
        entity, component = resolve_entity_by_id(self._entities, target_id)
        if entity is None:
            return None
        if isinstance(entity, PointMass):
            return self._to_3d(entity.position)
        if not isinstance(entity, RigidBody):
            return None
        if component is None:
            return np.asarray(entity.com_position, dtype=float)
        rotation = entity.rotation_matrix()
        return np.asarray(entity.com_position + rotation @ component.position_body, dtype=float)

    def _handle_mouse_press(self, ev) -> bool:
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return False
        if self._interaction_mode == "select":
            return False
        view_pos = ev.position() if hasattr(ev, "position") else ev.localPos()
        if self._interaction_mode == "move":
            axis = self._pick_gizmo_axis(view_pos)
            if axis is not None:
                return self._start_drag(axis, view_pos)
            com_pick = self._pick_rigid_body_com(view_pos)
            if com_pick is not None:
                self.entity_selected.emit(com_pick)
                return True
        if self._interaction_mode == "rotate":
            axis = self._pick_gizmo_ring(view_pos)
            if axis is not None:
                return self._start_drag(axis, view_pos)
        picked = self._pick_entity(view_pos)
        if picked is not None:
            self.entity_selected.emit(picked)
            return True
        return False

    def _handle_mouse_move(self, ev) -> bool:
        if not self._dragging:
            return False
        if not (ev.buttons() & QtCore.Qt.MouseButton.LeftButton):
            self._stop_drag()
            return True
        view_pos = ev.position() if hasattr(ev, "position") else ev.localPos()
        if self._drag_last_pos is None or self._drag_target_id is None:
            return True
        delta = view_pos - self._drag_last_pos
        if self._interaction_mode == "move":
            delta_pixels = delta.x() * self._drag_screen_dir[0] + delta.y() * self._drag_screen_dir[1]
            delta_world = self._drag_axis_dir * (delta_pixels * self._drag_scale)
            self.transform_requested.emit(self._drag_target_id, "move", delta_world)
        elif self._interaction_mode == "rotate":
            if self._drag_center_screen is None:
                angle_deg = delta.x() * self._rotate_scale_deg
            else:
                curr_vec = np.array(
                    [
                        view_pos.x() - self._drag_center_screen.x(),
                        view_pos.y() - self._drag_center_screen.y(),
                    ],
                    dtype=float,
                )
                if np.hypot(curr_vec[0], curr_vec[1]) <= 1e-6 or np.hypot(
                    self._drag_prev_vec[0], self._drag_prev_vec[1]
                ) <= 1e-6:
                    angle_deg = 0.0
                else:
                    prev_norm = self._drag_prev_vec / np.hypot(
                        self._drag_prev_vec[0], self._drag_prev_vec[1]
                    )
                    curr_norm = curr_vec / np.hypot(curr_vec[0], curr_vec[1])
                    cross = prev_norm[0] * curr_norm[1] - prev_norm[1] * curr_norm[0]
                    dot = prev_norm[0] * curr_norm[0] + prev_norm[1] * curr_norm[1]
                    angle_deg = float(np.degrees(np.arctan2(cross, dot)))
                    self._drag_prev_vec = curr_vec
            if abs(angle_deg) > 1e-6:
                self.transform_requested.emit(
                    self._drag_target_id,
                    "rotate",
                    {"axis": self._drag_axis_dir.copy(), "angle_deg": float(angle_deg)},
                )
        self._drag_last_pos = view_pos
        return True

    def _handle_mouse_release(self, ev) -> bool:
        if self._dragging:
            self._stop_drag()
            return True
        return False

    def _start_drag(self, axis: str, view_pos: QtCore.QPointF) -> bool:
        if self._gizmo_target_id is None:
            return False
        axis_dir = {
            "x": np.array([1.0, 0.0, 0.0], dtype=float),
            "y": np.array([0.0, 1.0, 0.0], dtype=float),
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
        }.get(axis)
        if axis_dir is None:
            return False
        self._dragging = True
        self._drag_axis = axis
        self._drag_target_id = self._gizmo_target_id
        self._drag_last_pos = view_pos
        self._drag_center_screen = self._screen_for_world(self._gizmo_pos)
        if self._drag_center_screen is not None:
            vec = np.array(
                [
                    view_pos.x() - self._drag_center_screen.x(),
                    view_pos.y() - self._drag_center_screen.y(),
                ],
                dtype=float,
            )
            if np.hypot(vec[0], vec[1]) > 1e-6:
                self._drag_prev_vec = vec
        self._drag_axis_dir = axis_dir
        if self._interaction_mode == "move":
            center = self._screen_for_world(self._gizmo_pos)
            end = self._screen_for_world(self._gizmo_pos + axis_dir * self._gizmo_axis_len)
            if center is None or end is None:
                self._drag_screen_dir = np.array([1.0, 0.0], dtype=float)
                self._drag_scale = 0.01
                return True
            screen_vec = np.array([end.x() - center.x(), end.y() - center.y()], dtype=float)
            screen_len = float(np.hypot(screen_vec[0], screen_vec[1]))
            if screen_len <= 1e-6:
                self._drag_screen_dir = np.array([1.0, 0.0], dtype=float)
                self._drag_scale = 0.01
            else:
                self._drag_screen_dir = screen_vec / screen_len
                self._drag_scale = self._gizmo_axis_len / screen_len
        return True

    def _stop_drag(self) -> None:
        self._dragging = False
        self._drag_axis = None
        self._drag_target_id = None
        self._drag_last_pos = None
        self._drag_center_screen = None

    def _pick_gizmo_axis(self, view_pos: QtCore.QPointF) -> str | None:
        if self._gizmo_target_id is None:
            return None
        center = self._screen_for_world(self._gizmo_pos)
        if center is None:
            return None
        threshold = 12.0
        best_axis = None
        best_dist = float("inf")
        for axis, axis_dir in (
            ("x", np.array([1.0, 0.0, 0.0], dtype=float)),
            ("y", np.array([0.0, 1.0, 0.0], dtype=float)),
            ("z", np.array([0.0, 0.0, 1.0], dtype=float)),
        ):
            end = self._screen_for_world(self._gizmo_pos + axis_dir * self._gizmo_axis_len)
            if end is None:
                continue
            dist = self._distance_to_segment(view_pos, center, end)
            screen_len = float(np.hypot(end.x() - center.x(), end.y() - center.y()))
            dynamic_threshold = max(threshold, screen_len * 0.8)
            if dist <= dynamic_threshold and dist < best_dist:
                best_dist = dist
                best_axis = axis
                continue
            if dist < best_dist:
                best_dist = dist
                best_axis = axis
        if best_dist <= threshold:
            return best_axis
        return None

    def _pick_gizmo_ring(self, view_pos: QtCore.QPointF) -> str | None:
        if self._gizmo_target_id is None:
            return None
        threshold = 8.0
        rings = {
            "x": self._build_ring_points(axis="x"),
            "y": self._build_ring_points(axis="y"),
            "z": self._build_ring_points(axis="z"),
        }
        best_axis = None
        best_dist = float("inf")
        for axis, points in rings.items():
            screen_points = []
            for point in points:
                screen = self._screen_for_world(point)
                if screen is None:
                    screen_points = []
                    break
                screen_points.append(screen)
            if len(screen_points) < 2:
                continue
            dist = self._distance_to_polyline(view_pos, screen_points)
            if dist < best_dist:
                best_dist = dist
                best_axis = axis
        if best_dist <= threshold:
            return best_axis
        return None

    def _pick_rigid_body_com(self, view_pos: QtCore.QPointF) -> str | None:
        if self._rb_com_positions.size == 0:
            return None
        viewport = (0, 0, self._viewport_size[0], self._viewport_size[1])
        projection = self._view.projectionMatrix(viewport, viewport)
        modelview = self._view.viewMatrix()
        min_distance = float("inf")
        picked_id: str | None = None
        for idx, pos in enumerate(self._rb_com_positions):
            screen_pos = self._project_point(pos, modelview, projection, viewport)
            if screen_pos is None:
                continue
            dist = np.hypot(screen_pos.x() - view_pos.x(), screen_pos.y() - view_pos.y())
            if dist < min_distance:
                min_distance = dist
                picked_id = self._rb_com_ids[idx]
        if min_distance <= self.PICK_RADIUS_PX:
            return picked_id
        return None

    def _build_ring_points(self, axis: str) -> np.ndarray:
        ring = self._gizmo_ring_radius
        points = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
        cos_vals = np.cos(points) * ring
        sin_vals = np.sin(points) * ring
        if axis == "x":
            base = np.column_stack((np.zeros_like(cos_vals), cos_vals, sin_vals))
        elif axis == "y":
            base = np.column_stack((cos_vals, np.zeros_like(cos_vals), sin_vals))
        else:
            base = np.column_stack((cos_vals, sin_vals, np.zeros_like(cos_vals)))
        return base + self._gizmo_pos.reshape(1, 3)

    def _screen_for_world(self, pos: np.ndarray) -> QtCore.QPointF | None:
        viewport = (0, 0, self._viewport_size[0], self._viewport_size[1])
        projection = self._view.projectionMatrix(viewport, viewport)
        modelview = self._view.viewMatrix()
        return self._project_point(pos, modelview, projection, viewport)

    @staticmethod
    def _distance_to_segment(p: QtCore.QPointF, a: QtCore.QPointF, b: QtCore.QPointF) -> float:
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        px, py = p.x(), p.y()
        dx = bx - ax
        dy = by - ay
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return float(np.hypot(px - ax, py - ay))
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return float(np.hypot(px - proj_x, py - proj_y))

    @classmethod
    def _distance_to_polyline(cls, p: QtCore.QPointF, points: list[QtCore.QPointF]) -> float:
        if len(points) < 2:
            return float("inf")
        min_dist = float("inf")
        for idx in range(len(points) - 1):
            dist = cls._distance_to_segment(p, points[idx], points[idx + 1])
            if dist < min_dist:
                min_dist = dist
        dist = cls._distance_to_segment(p, points[-1], points[0])
        return min(dist, min_dist)

    def set_orientation(self, key: str) -> None:
        azimuth, elevation = self._orientation_angles(key)
        center_vec = self._last_com if self._last_com.size == 3 else np.zeros(3, dtype=float)
        center = QtGui.QVector3D(float(center_vec[0]), float(center_vec[1]), float(center_vec[2]))
        distance = float(self._view.opts.get("distance", 20.0))
        self._animate_to_orientation(center, distance, azimuth, elevation)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_overlay()

    def _position_overlay(self) -> None:
        size = self._orientation_widget.sizeHint()
        x = max(self._view.width() - size.width() - self._orientation_margin, 0)
        y = self._orientation_margin
        self._orientation_widget.move(x, y)

    @staticmethod
    def _orientation_angles(key: str) -> tuple[float, float]:
        mapping = {
            "x": (0.0, 0.0),
            "-x": (180.0, 0.0),
            "y": (90.0, 0.0),
            "-y": (-90.0, 0.0),
            "z": (0.0, 90.0),
            "-z": (0.0, -90.0),
        }
        return mapping.get(key, (0.0, 0.0))

    def _orbit_from_gizmo(self, dx: float, dy: float) -> None:
        azimuth = float(self._view.opts.get("azimuth", 0.0))
        elevation = float(self._view.opts.get("elevation", 0.0))
        azimuth += dx * 0.6
        elevation -= dy * 0.6
        elevation = max(-89.0, min(89.0, elevation))
        center_vec = self._last_com if self._last_com.size == 3 else np.zeros(3, dtype=float)
        center = QtGui.QVector3D(float(center_vec[0]), float(center_vec[1]), float(center_vec[2]))
        distance = float(self._view.opts.get("distance", 20.0))
        self._view.setCameraPosition(pos=center, distance=distance, azimuth=azimuth, elevation=elevation)

    def _animate_to_orientation(
        self, center: QtGui.QVector3D, distance: float, azimuth: float, elevation: float
    ) -> None:
        current_center = self._view.opts.get("center", QtGui.QVector3D(0.0, 0.0, 0.0))
        current_distance = float(self._view.opts.get("distance", distance))
        current_azimuth = float(self._view.opts.get("azimuth", 0.0))
        current_elevation = float(self._view.opts.get("elevation", 0.0))
        self._orientation_anim = {
            "timer": QtCore.QElapsedTimer(),
            "duration_ms": 280,
            "start_center": current_center,
            "end_center": center,
            "start_distance": current_distance,
            "end_distance": distance,
            "start_azimuth": current_azimuth,
            "end_azimuth": azimuth,
            "start_elevation": current_elevation,
            "end_elevation": elevation,
        }
        self._orientation_anim["timer"].start()
        if not self._orientation_timer.isActive():
            self._orientation_timer.start()

    def _update_orientation_animation(self) -> None:
        if self._orientation_anim is None:
            self._orientation_timer.stop()
            return
        elapsed = self._orientation_anim["timer"].elapsed()
        duration = self._orientation_anim["duration_ms"]
        t = min(max(elapsed / duration, 0.0), 1.0)
        ease = t * t * (3.0 - 2.0 * t)
        center = self._lerp_vec(
            self._orientation_anim["start_center"],
            self._orientation_anim["end_center"],
            ease,
        )
        distance = self._lerp(
            self._orientation_anim["start_distance"],
            self._orientation_anim["end_distance"],
            ease,
        )
        azimuth = self._lerp_angle(
            self._orientation_anim["start_azimuth"],
            self._orientation_anim["end_azimuth"],
            ease,
        )
        elevation = self._lerp(
            self._orientation_anim["start_elevation"],
            self._orientation_anim["end_elevation"],
            ease,
        )
        self._view.setCameraPosition(
            pos=center, distance=float(distance), azimuth=float(azimuth), elevation=float(elevation)
        )
        if t >= 1.0:
            self._orientation_anim = None
            self._orientation_timer.stop()

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def _lerp_vec(a: QtGui.QVector3D, b: QtGui.QVector3D, t: float) -> QtGui.QVector3D:
        return QtGui.QVector3D(
            a.x() + (b.x() - a.x()) * t,
            a.y() + (b.y() - a.y()) * t,
            a.z() + (b.z() - a.z()) * t,
        )

    @staticmethod
    def _lerp_angle(a: float, b: float, t: float) -> float:
        delta = (b - a) % 360.0
        if delta > 180.0:
            delta -= 360.0
        return a + delta * t

    def _update_meshes(self, entities, display: DisplayOptions) -> None:
        if not entities or not display.show_mesh:
            for mesh_item in self._mesh_items.values():
                mesh_item.setVisible(False)
            return
        if not mesh_loading_available():
            for mesh_item in self._mesh_items.values():
                mesh_item.setVisible(False)
            return

        active_ids: set[str] = set()
        for entity in entities:
            if not isinstance(entity, RigidBody):
                continue
            if entity.mesh is None or not entity.mesh.path:
                continue
            active_ids.add(entity.entity_id)
            mesh_path = self._resolve_mesh_path(entity.mesh)
            if mesh_path is None:
                continue
            mesh_key = str(mesh_path)
            mesh_item = self._mesh_items.get(entity.entity_id)
            if mesh_item is None:
                mesh_item = gl.GLMeshItem(
                    meshdata=gl.MeshData(vertexes=np.zeros((0, 3), dtype=float), faces=np.zeros((0, 3), dtype=np.int32)),
                    smooth=False,
                    drawEdges=False,
                    drawFaces=True,
                    glOptions="translucent",
                )
                self._mesh_items[entity.entity_id] = mesh_item
                self._view.addItem(mesh_item)
            try:
                mesh_data = load_mesh_data(mesh_path)
            except Exception as exc:
                self._log.warning("Mesh load failed for %s: %s", mesh_path, exc)
                continue
            if self._mesh_sources.get(entity.entity_id) != mesh_key:
                self._mesh_sources[entity.entity_id] = mesh_key
            vertices_world = self._transform_mesh_vertices(mesh_data.vertices, entity, entity.mesh)
            use_model_colors = getattr(display, "mesh_color_mode", "gray") == "model"
            vertex_colors = None
            if use_model_colors:
                vertex_colors = self._normalize_vertex_colors(mesh_data.vertex_colors, 1.0)
            if vertex_colors is not None:
                mesh_item.setMeshData(
                    meshdata=gl.MeshData(vertexes=vertices_world, faces=mesh_data.faces, vertexColors=vertex_colors)
                )
            else:
                mesh_item.setMeshData(meshdata=gl.MeshData(vertexes=vertices_world, faces=mesh_data.faces))
                opacity = float(display.mesh_opacity) if not use_model_colors else 1.0
                color = (*self.MESH_BASE_COLOR[:3], opacity)
                mesh_item.setColor(color)
            mesh_item.setVisible(True)

        for entity_id in list(self._mesh_items.keys()):
            if entity_id not in active_ids:
                mesh_item = self._mesh_items.pop(entity_id)
                self._mesh_sources.pop(entity_id, None)
                self._view.removeItem(mesh_item)

    def _update_rigid_body_coms(self) -> None:
        self._rb_com_ids = []
        positions = []
        for entity in self._entities:
            if isinstance(entity, RigidBody):
                self._rb_com_ids.append(entity.entity_id)
                positions.append(self._to_3d(entity.com_position))
        if positions:
            self._rb_com_positions = np.stack(positions).astype(np.float32, copy=False)
        else:
            self._rb_com_positions = np.zeros((0, 3), dtype=np.float32)
        self._rb_com_item.setData(pos=self._rb_com_positions)

    def _resolve_mesh_path(self, mesh: MeshMetadata) -> Path | None:
        mesh_path = Path(mesh.path)
        if not mesh_path.is_absolute():
            mesh_path = (self._project_root / mesh_path).resolve()
        if not mesh_path.exists():
            self._log.warning("Mesh path not found: %s", mesh_path)
            return None
        return mesh_path

    def _transform_mesh_vertices(self, vertices: np.ndarray, body: RigidBody, mesh: MeshMetadata) -> np.ndarray:
        scaled = vertices * mesh.scale.reshape(1, 3)
        mesh_rotation = self._quat_to_matrix(mesh.rotation_body)
        rotated_local = (mesh_rotation @ scaled.T).T
        offset_local = rotated_local + mesh.offset_body.reshape(1, 3)
        body_rotation = body.rotation_matrix()
        world = (body_rotation @ offset_local.T).T + body.com_position.reshape(1, 3)
        return world.astype(np.float32, copy=False)

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = np.asarray(q, dtype=float).reshape(4)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    @staticmethod
    def _normalize_vertex_colors(colors: np.ndarray | None, opacity: float) -> np.ndarray | None:
        if colors is None or colors.size == 0:
            return None
        colors_arr = np.asarray(colors)
        if colors_arr.ndim != 2 or colors_arr.shape[1] not in (3, 4):
            return None
        colors_float = colors_arr.astype(np.float32, copy=True)
        if colors_float.max() > 1.0:
            colors_float /= 255.0
        if colors_float.shape[1] == 3:
            alpha = np.ones((colors_float.shape[0], 1), dtype=np.float32)
            colors_float = np.concatenate([colors_float, alpha], axis=1)
        colors_float[:, 3] *= float(opacity)
        return colors_float

    def _handle_click(self, view_pos: QtCore.QPointF) -> bool:
        picked = self._pick_entity(view_pos)
        if picked is not None:
            self.entity_selected.emit(picked)
            return True
        return False

    def _pick_entity(self, view_pos: QtCore.QPointF) -> str | None:
        if self._positions.size == 0:
            return None
        viewport = (0, 0, self._viewport_size[0], self._viewport_size[1])
        projection = self._view.projectionMatrix(viewport, viewport)
        modelview = self._view.viewMatrix()
        min_distance = float("inf")
        picked_id: str | None = None
        for idx, pos in enumerate(self._positions):
            screen_pos = self._project_point(pos, modelview, projection, viewport)
            if screen_pos is None:
                continue
            dist = np.hypot(screen_pos.x() - view_pos.x(), screen_pos.y() - view_pos.y())
            if dist < min_distance:
                min_distance = dist
                picked_id = self._entity_ids[idx]
        if min_distance <= self.PICK_RADIUS_PX:
            return picked_id
        return None

    @staticmethod
    def _project_point(
        pos: np.ndarray,
        modelview: QtGui.QMatrix4x4,
        projection: QtGui.QMatrix4x4,
        viewport: tuple[int, int, int, int],
    ) -> QtCore.QPointF | None:
        vec = QtGui.QVector4D(float(pos[0]), float(pos[1]), float(pos[2]), 1.0)
        clip = projection.map(modelview.map(vec))
        if clip.w() == 0.0:
            return None
        ndc_x = clip.x() / clip.w()
        ndc_y = clip.y() / clip.w()
        if ndc_x < -1.2 or ndc_x > 1.2 or ndc_y < -1.2 or ndc_y > 1.2:
            return None
        _, _, width, height = viewport
        screen_x = (ndc_x + 1.0) * 0.5 * width
        screen_y = (1.0 - ndc_y) * 0.5 * height
        return QtCore.QPointF(screen_x, screen_y)

    def _debug_enabled(self) -> bool:
        return os.getenv("SDW_DEBUG_3D", "").lower() in {"1", "true", "yes", "on"}
