from __future__ import annotations

from typing import List

import logging
import os

import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtGui, QtWidgets

from ...core.physics import FrameVectors, VectorSegment
from .base import OverlayOptions, Renderer


class Renderer3D(Renderer):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._log = logging.getLogger(__name__)
        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor("w")
        self._view.opts["distance"] = 20.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._axis_item = gl.GLAxisItem(size=QtGui.QVector3D(2.0, 2.0, 2.0))
        self._view.addItem(self._axis_item)

        self._points_item = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3)),
            size=7.0,
            pxMode=True,
            glOptions="opaque",
        )
        self._origin_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(0.38, 0.38, 0.38, 1.0),
            size=6.0,
            pxMode=True,
            glOptions="opaque",
        )
        self._com_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(0.83, 0.18, 0.18, 1.0),
            size=10.0,
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
        self._auto_fit_done = False

    def set_view_range(self, view_range: tuple[float, float, float, float]) -> None:
        _ = view_range

    def set_scene(self, frame_vectors: FrameVectors, overlays: OverlayOptions, selected_id: str | None) -> None:
        positions = self._to_3d_stack(frame_vectors.positions)
        self._positions = positions
        self._entity_ids = list(frame_vectors.entity_ids)

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
            sizes = 6.0 + 3.0 * np.sqrt(np.maximum(masses, 0.0))
            colors = np.tile(np.array([0.1, 0.46, 0.82, 1.0], dtype=np.float32), (positions.shape[0], 1))
            if selected_id in self._entity_ids:
                idx = self._entity_ids.index(selected_id)
                colors[idx] = np.array([1.0, 0.56, 0.0, 1.0], dtype=np.float32)
                sizes[idx] += 2.0
            self._points_item.setData(pos=positions.astype(np.float32, copy=False), size=sizes, color=colors)

        origin_pos = self._to_3d(frame_vectors.origin)
        com_pos = self._to_3d(frame_vectors.com)
        self._origin_item.setData(pos=origin_pos.reshape(1, 3))
        self._com_item.setData(pos=com_pos.reshape(1, 3))

        self._set_vector_item(self._r_op_item, frame_vectors.r_op_segments, overlays.show_r_op)
        self._set_vector_item(self._r_cp_item, frame_vectors.r_cp_segments, overlays.show_r_cp)
        self._set_vector_item(self._r_oc_item, [frame_vectors.r_oc_segment], overlays.show_r_oc)

        if not self._auto_fit_done and positions.size:
            self._auto_fit_camera(positions, com_pos)

    def clear(self) -> None:
        self._points_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._origin_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._com_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._r_op_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._r_cp_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._r_oc_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self._auto_fit_done = False

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

    def _auto_fit_camera(self, positions: np.ndarray, center_vec: np.ndarray) -> None:
        min_vals = positions.min(axis=0)
        max_vals = positions.max(axis=0)
        span = np.maximum(max_vals - min_vals, 1e-6)
        radius = 0.5 * float(np.linalg.norm(span))
        distance = max(5.0, radius * 3.0)
        center = QtGui.QVector3D(float(center_vec[0]), float(center_vec[1]), float(center_vec[2]))
        self._view.setCameraPosition(pos=center, distance=distance)
        self._auto_fit_done = True

    def _debug_enabled(self) -> bool:
        return os.getenv("SDW_DEBUG_3D", "").lower() in {"1", "true", "yes", "on"}
