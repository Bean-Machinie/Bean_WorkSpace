from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class SimulationPanel(QtWidgets.QGroupBox):
    settings_changed = QtCore.Signal(float, str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Simulation", parent)
        layout = QtWidgets.QFormLayout(self)

        self._dt_spin = QtWidgets.QDoubleSpinBox()
        self._dt_spin.setRange(1e-6, 1e6)
        self._dt_spin.setDecimals(6)
        self._dt_spin.setValue(0.05)

        self._integrator_combo = QtWidgets.QComboBox()
        self._integrator_combo.addItem("Symplectic Euler", "symplectic_euler")

        layout.addRow("dt (s)", self._dt_spin)
        layout.addRow("Integrator", self._integrator_combo)

        self._dt_spin.valueChanged.connect(self._emit_settings_changed)
        self._integrator_combo.currentIndexChanged.connect(self._emit_settings_changed)

    def set_settings(self, dt: float, integrator: str) -> None:
        self._dt_spin.blockSignals(True)
        self._integrator_combo.blockSignals(True)
        self._dt_spin.setValue(float(dt))
        idx = self._integrator_combo.findData(integrator)
        if idx >= 0:
            self._integrator_combo.setCurrentIndex(idx)
        self._dt_spin.blockSignals(False)
        self._integrator_combo.blockSignals(False)

    def _emit_settings_changed(self) -> None:
        self.settings_changed.emit(float(self._dt_spin.value()), str(self._integrator_combo.currentData()))
