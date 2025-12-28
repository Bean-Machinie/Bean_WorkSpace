from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ...core.physics import FrameChoice
from ..rendering.base import OverlayOptions


class ViewOptionsPanel(QtWidgets.QGroupBox):
    frame_changed = QtCore.Signal(object)
    overlays_changed = QtCore.Signal(object)
    renderer_changed = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("View Options", parent)
        layout = QtWidgets.QFormLayout(self)

        self._frame_combo = QtWidgets.QComboBox()
        self._frame_combo.addItem("World (O)", FrameChoice.WORLD)
        self._frame_combo.addItem("Center of Mass (C)", FrameChoice.COM)

        self._renderer_combo = QtWidgets.QComboBox()
        self._renderer_combo.addItem("2D (stable)", "2d")
        self._renderer_combo.addItem("3D (experimental)", "3d")
        self._renderer_combo.setCurrentIndex(1)

        self._show_r_op = QtWidgets.QCheckBox("Show r_OP")
        self._show_r_oc = QtWidgets.QCheckBox("Show r_OC")
        self._show_r_cp = QtWidgets.QCheckBox("Show r_CP")
        self._show_r_op.setChecked(False)
        self._show_r_oc.setChecked(False)
        self._show_r_cp.setChecked(True)

        self._show_grid_xy = QtWidgets.QCheckBox("Show Grid (XY)")
        self._show_grid_xz = QtWidgets.QCheckBox("Show Grid (XZ)")
        self._show_grid_yz = QtWidgets.QCheckBox("Show Grid (YZ)")
        self._show_grid_xy.setChecked(True)
        self._show_grid_xz.setChecked(False)
        self._show_grid_yz.setChecked(False)

        layout.addRow("Reference Frame", self._frame_combo)
        layout.addRow("Renderer", self._renderer_combo)
        layout.addRow(self._show_r_op)
        layout.addRow(self._show_r_oc)
        layout.addRow(self._show_r_cp)
        layout.addRow(QtWidgets.QLabel("3D View"))
        layout.addRow(self._show_grid_xy)
        layout.addRow(self._show_grid_xz)
        layout.addRow(self._show_grid_yz)

        self._frame_combo.currentIndexChanged.connect(self._emit_frame)
        self._renderer_combo.currentIndexChanged.connect(self._emit_renderer)
        self._show_r_op.toggled.connect(self._emit_overlays)
        self._show_r_oc.toggled.connect(self._emit_overlays)
        self._show_r_cp.toggled.connect(self._emit_overlays)
        self._show_grid_xy.toggled.connect(self._emit_overlays)
        self._show_grid_xz.toggled.connect(self._emit_overlays)
        self._show_grid_yz.toggled.connect(self._emit_overlays)

    def current_frame(self) -> FrameChoice:
        return self._frame_combo.currentData()

    def overlay_options(self) -> OverlayOptions:
        return OverlayOptions(
            show_r_op=self._show_r_op.isChecked(),
            show_r_oc=self._show_r_oc.isChecked(),
            show_r_cp=self._show_r_cp.isChecked(),
            show_grid_xy=self._show_grid_xy.isChecked(),
            show_grid_xz=self._show_grid_xz.isChecked(),
            show_grid_yz=self._show_grid_yz.isChecked(),
        )

    def set_renderer_availability(self, available: bool, message: str | None = None) -> None:
        model = self._renderer_combo.model()
        item = model.item(1)
        if item is not None:
            item.setEnabled(available)
        if not available:
            self._renderer_combo.setCurrentIndex(0)
            tooltip = message or "Install 3D extras (PyOpenGL) to enable 3D mode."
            self._renderer_combo.setToolTip(tooltip)
        else:
            self._renderer_combo.setToolTip("")

    def _emit_frame(self) -> None:
        self.frame_changed.emit(self.current_frame())

    def _emit_renderer(self) -> None:
        self.renderer_changed.emit(self._renderer_combo.currentData())

    def _emit_overlays(self) -> None:
        self.overlays_changed.emit(self.overlay_options())
