from __future__ import annotations

from PySide6 import QtWidgets


class ScenarioSettingsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        name: str,
        dt: float,
        integrator: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scenario Settings")
        self.resize(360, 220)

        layout = QtWidgets.QFormLayout(self)

        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setText(name)

        self._dt_spin = QtWidgets.QDoubleSpinBox()
        self._dt_spin.setRange(1e-6, 1e6)
        self._dt_spin.setDecimals(6)
        self._dt_spin.setValue(float(dt))

        self._integrator_combo = QtWidgets.QComboBox()
        self._integrator_combo.addItem("Symplectic Euler", "symplectic_euler")
        idx = self._integrator_combo.findData(integrator)
        if idx >= 0:
            self._integrator_combo.setCurrentIndex(idx)

        layout.addRow("Scenario Name", self._name_edit)
        layout.addRow("dt (s)", self._dt_spin)
        layout.addRow("Integrator", self._integrator_combo)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self) -> tuple[str, float, str]:
        return (
            self._name_edit.text(),
            float(self._dt_spin.value()),
            str(self._integrator_combo.currentData()),
        )
