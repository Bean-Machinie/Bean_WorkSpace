from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...core.model import PointMass


class SceneView(QtWidgets.QWidget):
    entity_selected = QtCore.Signal(str)
    entity_dragged = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot = pg.PlotWidget(background="w")
        self._plot.setAspectLocked(True)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "X")
        self._plot.setLabel("left", "Y")

        self._points = pg.ScatterPlotItem()
        self._com_marker = pg.ScatterPlotItem(symbol="x", size=14, pen=pg.mkPen("#d32f2f", width=2))
        self._plot.addItem(self._points)
        self._plot.addItem(self._com_marker)

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

    def update_scene(self, entities: Iterable[PointMass], com: np.ndarray | None, selected_id: str | None) -> None:
        entities_list = list(entities)
        self._entity_ids = [entity.entity_id for entity in entities_list]
        self._positions = np.stack([entity.position for entity in entities_list]) if entities_list else np.zeros((0, 2))
        self._selected_id = selected_id

        spots = []
        for entity in entities_list:
            size = 6.0 + 3.0 * np.sqrt(entity.mass)
            is_selected = entity.entity_id == selected_id
            brush = pg.mkBrush("#1976d2" if not is_selected else "#ff8f00")
            pen = pg.mkPen("#0d47a1" if not is_selected else "#e65100", width=2)
            spots.append({"pos": entity.position, "size": size, "brush": brush, "pen": pen})
        self._points.setData(spots)
        if com is None or not entities_list:
            com_pos = np.empty((0, 2), dtype=float)
        else:
            # pyqtgraph expects positions as an (N, 2) array via pos=.
            com_pos = np.asarray(com, dtype=float).reshape(1, 2)
        self._com_marker.setData(pos=com_pos)

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
