from __future__ import annotations

from PySide6 import QtWidgets


class NewScenarioDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Scenario")
        self.resize(360, 200)

        layout = QtWidgets.QFormLayout(self)

        self._name_edit = QtWidgets.QLineEdit()
        self._name_edit.setText("Untitled Scenario")

        self._dt_spin = QtWidgets.QDoubleSpinBox()
        self._dt_spin.setRange(1e-6, 1e6)
        self._dt_spin.setDecimals(6)
        self._dt_spin.setValue(0.05)

        self._integrator_combo = QtWidgets.QComboBox()
        self._integrator_combo.addItem("Symplectic Euler", "symplectic_euler")

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
