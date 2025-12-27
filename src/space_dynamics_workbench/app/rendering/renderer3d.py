from __future__ import annotations

from pathlib import Path
from typing import List

import logging
import os

import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui, QtWidgets

from ...core.model import MeshMetadata, RigidBody
from ...core.physics import FrameVectors, VectorSegment
from ..mesh_loading import load_mesh_data, mesh_loading_available
from .base import DisplayOptions, OverlayOptions, Renderer


class _PickingGLViewWidget(gl.GLViewWidget):
    def __init__(self, on_click, parent=None) -> None:
        super().__init__(parent=parent)
        self._on_click = on_click
        self._press_pos: QtCore.QPointF | None = None

    def mousePressEvent(self, ev) -> None:
        self._press_pos = ev.position() if hasattr(ev, "position") else ev.localPos()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        release_pos = ev.position() if hasattr(ev, "position") else ev.localPos()
        if self._press_pos is not None and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if (release_pos - self._press_pos).manhattanLength() < 4.0:
                self._on_click(release_pos)
        self._press_pos = None
        super().mouseReleaseEvent(ev)


class Renderer3D(Renderer):
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
        self._view = _PickingGLViewWidget(self._handle_click)
        self._view.setBackgroundColor(self.BG_COLOR)
        self._view.opts["distance"] = 20.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._axis_item = gl.GLAxisItem(size=QtGui.QVector3D(6.0, 6.0, 6.0))
        self._view.addItem(self._axis_item)
        self._axis_lines = self._make_axis_lines()
        for axis_line in self._axis_lines:
            self._view.addItem(axis_line)

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
        self._view.addItem(self._r_op_item)
        self._view.addItem(self._r_cp_item)
        self._view.addItem(self._r_oc_item)

        self._entity_ids: List[str] = []
        self._positions: np.ndarray = np.zeros((0, 3), dtype=float)
        self._last_com: np.ndarray = np.zeros(3, dtype=float)
        self._viewport_size: tuple[int, int] = (1, 1)
        self._mesh_items: dict[str, gl.GLMeshItem] = {}
        self._mesh_sources: dict[str, str] = {}
        self._project_root = Path.cwd()

    def set_view_range(self, view_range: tuple[float, float, float, float]) -> None:
        _ = view_range

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

        self._set_vector_item(self._r_op_item, frame_vectors.r_op_segments, overlays.show_r_op)
        self._set_vector_item(self._r_cp_item, frame_vectors.r_cp_segments, overlays.show_r_cp)
        self._set_vector_item(self._r_oc_item, [frame_vectors.r_oc_segment], overlays.show_r_oc)

        self._grid_xy_minor.setVisible(overlays.show_grid_xy)
        self._grid_xy_major.setVisible(overlays.show_grid_xy)
        self._grid_xz_minor.setVisible(overlays.show_grid_xz)
        self._grid_xz_major.setVisible(overlays.show_grid_xz)
        self._grid_yz_minor.setVisible(overlays.show_grid_yz)
        self._grid_yz_major.setVisible(overlays.show_grid_yz)

        self._update_meshes(entities, display_options)

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
        axis_len = 6.0
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
                    width=2.0,
                    antialias=True,
                    mode="lines",
                    glOptions="opaque",
                )
            )
        return items

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
            mesh_item.setMeshData(meshdata=gl.MeshData(vertexes=vertices_world, faces=mesh_data.faces))
            color = (*self.MESH_BASE_COLOR[:3], float(display.mesh_opacity))
            mesh_item.setColor(color)
            mesh_item.setVisible(True)

        for entity_id in list(self._mesh_items.keys()):
            if entity_id not in active_ids:
                mesh_item = self._mesh_items.pop(entity_id)
                self._mesh_sources.pop(entity_id, None)
                self._view.removeItem(mesh_item)

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

    def _handle_click(self, view_pos: QtCore.QPointF) -> None:
        picked = self._pick_entity(view_pos)
        if picked is not None:
            self.entity_selected.emit(picked)

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
