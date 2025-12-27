from __future__ import annotations

from typing import List, Optional

import enum

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...core.physics import FrameVectors, VectorSegment
from ..rendering.base import OverlayOptions, Renderer


class SceneView(Renderer):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._space_pan_active = False
        self._view_box = _SceneViewBox(self._is_space_pan_active)
        self._plot = pg.PlotWidget(background="w", viewBox=self._view_box)
        self._plot.setMouseTracking(True)
        self._plot.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._plot.viewport().setMouseTracking(True)
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "X")
        self._plot.setLabel("left", "Y")
        self._points = pg.ScatterPlotItem()
        self._com_marker = pg.ScatterPlotItem(symbol="x", size=14, pen=pg.mkPen("#d32f2f", width=2))
        self._origin_marker = pg.ScatterPlotItem(symbol="o", size=8, pen=pg.mkPen("#616161", width=2))
        self._r_op_item = pg.PlotDataItem(pen=pg.mkPen("#1976d2", width=1.5))
        self._r_cp_item = pg.PlotDataItem(pen=pg.mkPen("#388e3c", width=1.5))
        self._r_oc_item = pg.PlotDataItem(pen=pg.mkPen("#d32f2f", width=1.5))
        self._plot.addItem(self._points)
        self._plot.addItem(self._origin_marker)
        self._plot.addItem(self._com_marker)
        self._plot.addItem(self._r_op_item)
        self._plot.addItem(self._r_cp_item)
        self._plot.addItem(self._r_oc_item)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

        self._entity_ids: List[str] = []
        self._positions: np.ndarray = np.zeros((0, 2), dtype=float)
        self._masses: np.ndarray = np.zeros((0,), dtype=float)
        self._selected_id: Optional[str] = None
        self._hover_id: Optional[str] = None
        self._active_drag_id: Optional[str] = None
        self._pick_radius_ratio: float = 0.03

        self._interaction = _SceneInteractionController(self)

    def set_view_range(self, view_range: tuple[float, float, float, float]) -> None:
        x_min, x_max, y_min, y_max = view_range
        self._plot.setXRange(x_min, x_max, padding=0.0)
        self._plot.setYRange(y_min, y_max, padding=0.0)

    def set_scene(self, frame_vectors: FrameVectors, overlays: OverlayOptions, selected_id: str | None) -> None:
        positions = list(frame_vectors.positions)
        self._entity_ids = list(frame_vectors.entity_ids)
        self._masses = np.asarray(frame_vectors.masses, dtype=float) if frame_vectors.masses else np.zeros((0,), dtype=float)
        if positions:
            self._positions = np.stack([pos[:2] for pos in positions])
        else:
            self._positions = np.zeros((0, 2), dtype=float)
        self._selected_id = selected_id

        self._refresh_points()

        self._origin_marker.setData(pos=np.asarray(frame_vectors.origin[:2], dtype=float).reshape(1, 2))
        self._com_marker.setData(pos=np.asarray(frame_vectors.com[:2], dtype=float).reshape(1, 2))

        self._set_vector_item(self._r_op_item, frame_vectors.r_op_segments, overlays.show_r_op)
        self._set_vector_item(self._r_cp_item, frame_vectors.r_cp_segments, overlays.show_r_cp)
        self._set_vector_item(self._r_oc_item, [frame_vectors.r_oc_segment], overlays.show_r_oc)

    def clear(self) -> None:
        self._points.setData([])
        self._origin_marker.setData(pos=np.empty((0, 2), dtype=float))
        self._com_marker.setData(pos=np.empty((0, 2), dtype=float))
        self._r_op_item.setData([], [])
        self._r_cp_item.setData([], [])
        self._r_oc_item.setData([], [])
        self._hover_id = None
        self._active_drag_id = None

    def _set_vector_item(self, item: pg.PlotDataItem, segments: List[VectorSegment], visible: bool) -> None:
        item.setVisible(visible)
        if not visible or not segments:
            item.setData([], [])
            return
        x_vals: List[float] = []
        y_vals: List[float] = []
        for segment in segments:
            x_vals.extend([segment.start[0], segment.end[0], np.nan])
            y_vals.extend([segment.start[1], segment.end[1], np.nan])
        item.setData(x_vals, y_vals)

    def _refresh_points(self) -> None:
        spots = []
        for idx, pos in enumerate(self._positions):
            entity_id = self._entity_ids[idx] if idx < len(self._entity_ids) else None
            mass = float(self._masses[idx]) if idx < self._masses.size else 1.0
            size, brush, pen = self._point_style(
                mass,
                is_selected=entity_id == self._selected_id,
                is_hover=entity_id == self._hover_id,
                is_active=entity_id == self._active_drag_id,
            )
            spots.append({"pos": pos[:2], "size": size, "brush": brush, "pen": pen})
        self._points.setData(spots)

    def _point_style(
        self,
        mass: float,
        *,
        is_selected: bool,
        is_hover: bool,
        is_active: bool,
    ) -> tuple[float, pg.QtGui.QBrush, pg.QtGui.QPen]:
        base_size = 6.0 + 3.0 * np.sqrt(max(mass, 0.0))
        size = base_size + (2.0 if is_hover else 0.0) + (3.0 if is_active else 0.0)
        if is_active:
            brush = pg.mkBrush("#ffd54f")
            pen = pg.mkPen("#f57c00", width=2.5)
        elif is_selected:
            brush = pg.mkBrush("#ff8f00")
            pen = pg.mkPen("#e65100", width=2.0)
        elif is_hover:
            brush = pg.mkBrush("#64b5f6")
            pen = pg.mkPen("#1e88e5", width=2.0)
        else:
            brush = pg.mkBrush("#1976d2")
            pen = pg.mkPen("#0d47a1", width=2.0)
        return size, brush, pen

    def _set_hover_id(self, entity_id: Optional[str]) -> None:
        if self._hover_id == entity_id:
            return
        self._hover_id = entity_id
        self._refresh_points()

    def _set_active_drag_id(self, entity_id: Optional[str]) -> None:
        if self._active_drag_id == entity_id:
            return
        self._active_drag_id = entity_id
        self._refresh_points()

    def _pick_entity(self, view_pos: np.ndarray) -> Optional[str]:
        if self._positions.size == 0:
            return None
        distances = np.linalg.norm(self._positions - view_pos, axis=1)
        view_range = self._plot.getViewBox().viewRange()
        threshold = self._pick_radius_ratio * max(
            view_range[0][1] - view_range[0][0],
            view_range[1][1] - view_range[1][0],
        )
        idx = int(np.argmin(distances))
        if distances[idx] <= threshold:
            return self._entity_ids[idx]
        return None

    def _emit_entity_dragged(self, entity_id: str, new_pos: np.ndarray) -> None:
        self.entity_dragged.emit(entity_id, new_pos)

    def _emit_entity_selected(self, entity_id: str) -> None:
        self.entity_selected.emit(entity_id)

    def _set_cursor(self, cursor: QtCore.Qt.CursorShape) -> None:
        self._plot.setCursor(cursor)

    def _view_pos_from_event(self, event: QtCore.QEvent) -> QtCore.QPointF:
        scene_pos = self._plot.mapToScene(event.pos())
        return self._plot.getViewBox().mapSceneToView(scene_pos)

    def _is_space_pan_active(self) -> bool:
        return self._space_pan_active


class _InteractionMode(enum.Enum):
    NONE = enum.auto()
    DRAG_POINT = enum.auto()


class _SceneViewBox(pg.ViewBox):
    def __init__(self, allow_pan_fn, **kwargs) -> None:
        super().__init__(**kwargs)
        self._allow_pan_fn = allow_pan_fn

    def mouseDragEvent(self, ev, axis=None) -> None:
        if not self._allow_pan_fn():
            ev.ignore()
            return
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        super().mouseDragEvent(ev, axis=axis)


class _SceneInteractionController(QtCore.QObject):
    def __init__(self, view: SceneView) -> None:
        super().__init__(view)
        self._view = view
        self._plot = view._plot
        self._view_box = view._plot.getViewBox()
        self._mode = _InteractionMode.NONE
        self._space_pressed = False
        self._ctrl_pressed = False
        self._is_panning = False
        self._last_scene_pos: Optional[QtCore.QPointF] = None

        self._plot.installEventFilter(self)
        self._plot.viewport().installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        event_type = event.type()
        if event_type == QtCore.QEvent.KeyPress and obj is self._plot:
            return self._handle_key_press(event)
        if event_type == QtCore.QEvent.KeyRelease and obj is self._plot:
            return self._handle_key_release(event)
        if event_type == QtCore.QEvent.MouseButtonPress and obj is self._plot.viewport():
            return self._handle_mouse_press(event)
        if event_type == QtCore.QEvent.MouseMove and obj is self._plot.viewport():
            return self._handle_mouse_move(event)
        if event_type == QtCore.QEvent.MouseButtonRelease and obj is self._plot.viewport():
            return self._handle_mouse_release(event)
        if event_type == QtCore.QEvent.Leave and obj is self._plot.viewport():
            return self._handle_mouse_leave(event)
        return super().eventFilter(obj, event)

    def _handle_key_press(self, event: QtCore.QEvent) -> bool:
        if event.isAutoRepeat():
            return False
        if event.key() == QtCore.Qt.Key_Space:
            self._space_pressed = True
            if self._mode == _InteractionMode.DRAG_POINT:
                self._end_drag()
            self._view._set_hover_id(None)
            self._view._space_pan_active = True
            self._update_cursor()
            return True
        self._ctrl_pressed = bool(event.modifiers() & QtCore.Qt.ControlModifier)
        if self._ctrl_pressed and not self._space_pressed and self._last_scene_pos is not None:
            self._update_hover_from_scene_pos(self._last_scene_pos)
        self._update_cursor()
        return False

    def _handle_key_release(self, event: QtCore.QEvent) -> bool:
        if event.isAutoRepeat():
            return False
        if event.key() == QtCore.Qt.Key_Space:
            self._space_pressed = False
            self._is_panning = False
            self._view._space_pan_active = False
            self._update_cursor()
            return True
        self._ctrl_pressed = bool(event.modifiers() & QtCore.Qt.ControlModifier)
        if not self._ctrl_pressed:
            if self._mode == _InteractionMode.DRAG_POINT:
                self._end_drag()
            self._view._set_hover_id(None)
        self._update_cursor()
        return False

    def _handle_mouse_press(self, event: QtCore.QEvent) -> bool:
        if event.button() != QtCore.Qt.LeftButton:
            return False
        self._plot.setFocus()
        self._ctrl_pressed = bool(event.modifiers() & QtCore.Qt.ControlModifier)
        if self._space_pressed:
            self._is_panning = True
            self._view._set_cursor(QtCore.Qt.ClosedHandCursor)
            return False
        if self._ctrl_pressed:
            picked = self._pick_at_event(event)
            if picked is None:
                self._view._set_hover_id(None)
                self._update_cursor()
                return True
            self._start_drag(event, picked)
            return True
        picked = self._pick_at_event(event)
        if picked is not None:
            self._view._emit_entity_selected(picked)
        return True

    def _handle_mouse_move(self, event: QtCore.QEvent) -> bool:
        self._ctrl_pressed = bool(event.modifiers() & QtCore.Qt.ControlModifier)
        self._last_scene_pos = self._plot.mapToScene(event.pos())
        if self._space_pressed:
            if not (event.buttons() & QtCore.Qt.LeftButton):
                self._is_panning = False
            self._update_cursor()
            return False
        if self._mode == _InteractionMode.DRAG_POINT:
            if not (event.buttons() & QtCore.Qt.LeftButton):
                self._end_drag()
                return True
            self._update_drag(event)
            return True
        if self._space_pressed:
            self._view._set_hover_id(None)
            self._update_cursor()
            return False
        if self._ctrl_pressed:
            self._update_hover_from_scene_pos(self._last_scene_pos)
            self._update_cursor()
            return False
        if self._view._hover_id is not None:
            self._view._set_hover_id(None)
        self._update_cursor()
        return False

    def _handle_mouse_release(self, event: QtCore.QEvent) -> bool:
        if event.button() != QtCore.Qt.LeftButton:
            return False
        if self._space_pressed:
            self._is_panning = False
            self._update_cursor()
            return False
        if self._mode == _InteractionMode.DRAG_POINT:
            self._end_drag()
            return True
        self._update_cursor()
        return False

    def _handle_mouse_leave(self, event: QtCore.QEvent) -> bool:
        if self._mode == _InteractionMode.NONE:
            self._view._set_hover_id(None)
            self._view._set_cursor(QtCore.Qt.ArrowCursor)
        return False

    def _start_drag(self, event: QtCore.QEvent, entity_id: str) -> None:
        self._mode = _InteractionMode.DRAG_POINT
        self._view._set_active_drag_id(entity_id)
        self._view._set_hover_id(entity_id)
        self._view._emit_entity_selected(entity_id)
        self._view._set_cursor(QtCore.Qt.ClosedHandCursor)

    def _update_drag(self, event: QtCore.QEvent) -> None:
        if self._view._active_drag_id is None:
            return
        view_pos = self._view._view_pos_from_event(event)
        new_pos = np.array([view_pos.x(), view_pos.y()], dtype=float)
        self._view._emit_entity_dragged(self._view._active_drag_id, new_pos)
        self._view._set_cursor(QtCore.Qt.ClosedHandCursor)

    def _end_drag(self) -> None:
        self._mode = _InteractionMode.NONE
        self._view._set_active_drag_id(None)
        self._update_cursor()

    def _pick_at_event(self, event: QtCore.QEvent) -> Optional[str]:
        view_pos = self._view._view_pos_from_event(event)
        return self._view._pick_entity(np.array([view_pos.x(), view_pos.y()], dtype=float))

    def _update_hover_from_scene_pos(self, scene_pos: QtCore.QPointF) -> None:
        view_pos = self._view_box.mapSceneToView(scene_pos)
        picked = self._view._pick_entity(np.array([view_pos.x(), view_pos.y()], dtype=float))
        self._view._set_hover_id(picked)

    def _update_cursor(self) -> None:
        if self._mode == _InteractionMode.DRAG_POINT:
            self._view._set_cursor(QtCore.Qt.ClosedHandCursor)
            return
        if self._is_panning:
            self._view._set_cursor(QtCore.Qt.ClosedHandCursor)
            return
        if self._space_pressed:
            self._view._set_cursor(QtCore.Qt.OpenHandCursor)
            return
        if self._ctrl_pressed and self._view._hover_id is not None:
            self._view._set_cursor(QtCore.Qt.OpenHandCursor)
            return
        self._view._set_cursor(QtCore.Qt.ArrowCursor)
