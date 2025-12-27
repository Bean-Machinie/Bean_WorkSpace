from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from ...core.model import PointMass


class InspectorPanel(QtWidgets.QGroupBox):
    entity_updated = QtCore.Signal(str, float, object, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Inspector", parent)
        self._entity_id: Optional[str] = None
        self._block_updates = False

        layout = QtWidgets.QFormLayout(self)
        self._id_field = QtWidgets.QLineEdit()
        self._id_field.setReadOnly(True)

        self._mass_field = QtWidgets.QDoubleSpinBox()
        self._mass_field.setRange(1e-6, 1e9)
        self._mass_field.setDecimals(6)

        self._pos_x = self._make_spin()
        self._pos_y = self._make_spin()
        self._vel_x = self._make_spin()
        self._vel_y = self._make_spin()

        layout.addRow("ID", self._id_field)
        layout.addRow("Mass", self._mass_field)
        layout.addRow("Pos X", self._pos_x)
        layout.addRow("Pos Y", self._pos_y)
        layout.addRow("Vel X", self._vel_x)
        layout.addRow("Vel Y", self._vel_y)

        self._mass_field.valueChanged.connect(self._emit_update)
        self._pos_x.valueChanged.connect(self._emit_update)
        self._pos_y.valueChanged.connect(self._emit_update)
        self._vel_x.valueChanged.connect(self._emit_update)
        self._vel_y.valueChanged.connect(self._emit_update)

        self.setEnabled(False)

    def _make_spin(self) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(6)
        return spin

    def set_entity(self, entity: PointMass | None) -> None:
        self._block_updates = True
        if entity is None:
            self._entity_id = None
            self._id_field.setText("")
            self.setEnabled(False)
        else:
            self._entity_id = entity.entity_id
            self._id_field.setText(entity.entity_id)
            self._mass_field.setValue(entity.mass)
            self._pos_x.setValue(float(entity.position[0]))
            self._pos_y.setValue(float(entity.position[1]))
            self._vel_x.setValue(float(entity.velocity[0]))
            self._vel_y.setValue(float(entity.velocity[1]))
            self.setEnabled(True)
        self._block_updates = False

    def _emit_update(self) -> None:
        if self._block_updates or self._entity_id is None:
            return
        self.entity_updated.emit(
            self._entity_id,
            float(self._mass_field.value()),
            (float(self._pos_x.value()), float(self._pos_y.value())),
            (float(self._vel_x.value()), float(self._vel_y.value())),
        )
