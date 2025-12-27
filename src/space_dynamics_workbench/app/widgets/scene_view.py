from __future__ import annotations

from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...core.physics import FrameVectors, VectorSegment
from ..rendering.base import OverlayOptions, Renderer


class SceneView(Renderer):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot = pg.PlotWidget(background="w")
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
        self._selected_id: Optional[str] = None
        self._dragging_id: Optional[str] = None

        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def set_view_range(self, view_range: tuple[float, float, float, float]) -> None:
        x_min, x_max, y_min, y_max = view_range
        self._plot.setXRange(x_min, x_max, padding=0.0)
        self._plot.setYRange(y_min, y_max, padding=0.0)

    def set_scene(self, frame_vectors: FrameVectors, overlays: OverlayOptions, selected_id: str | None) -> None:
        positions = list(frame_vectors.positions)
        self._entity_ids = list(frame_vectors.entity_ids)
        if positions:
            self._positions = np.stack([pos[:2] for pos in positions])
        else:
            self._positions = np.zeros((0, 2))
        self._selected_id = selected_id

        spots = []
        for idx, pos in enumerate(positions):
            entity_id = self._entity_ids[idx] if idx < len(self._entity_ids) else None
            is_selected = entity_id == selected_id
            brush = pg.mkBrush("#1976d2" if not is_selected else "#ff8f00")
            pen = pg.mkPen("#0d47a1" if not is_selected else "#e65100", width=2)
            mass = frame_vectors.masses[idx] if idx < len(frame_vectors.masses) else 1.0
            size = 6.0 + 3.0 * np.sqrt(max(mass, 0.0))
            spots.append({"pos": pos[:2], "size": size, "brush": brush, "pen": pen})
        self._points.setData(spots)

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

    def _on_mouse_clicked(self, event: pg.GraphicsScene.mouseEvents.MouseClickEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        clicked = self._plot.getViewBox().mapSceneToView(event.scenePos())
        if self._positions.size == 0:
            return
        click_pos = np.array([clicked.x(), clicked.y()], dtype=float)
        distances = np.linalg.norm(self._positions - click_pos, axis=1)
        view_range = self._plot.getViewBox().viewRange()
        threshold = 0.03 * max(view_range[0][1] - view_range[0][0], view_range[1][1] - view_range[1][0])
        idx = int(np.argmin(distances))
        if distances[idx] <= threshold:
            entity_id = self._entity_ids[idx]
            self._dragging_id = entity_id
            self.entity_selected.emit(entity_id)

    def _on_mouse_moved(self, pos: QtCore.QPointF) -> None:
        if self._dragging_id is None:
            return
        if not (QtWidgets.QApplication.mouseButtons() & QtCore.Qt.LeftButton):
            self._dragging_id = None
            return
        moved = self._plot.getViewBox().mapSceneToView(pos)
        new_pos = np.array([moved.x(), moved.y()], dtype=float)
        self.entity_dragged.emit(self._dragging_id, new_pos)
