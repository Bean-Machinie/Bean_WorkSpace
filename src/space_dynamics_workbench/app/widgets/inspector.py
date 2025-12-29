from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from ...core.model import PointMass, RigidBody


class InspectorPanel(QtWidgets.QGroupBox):
    point_mass_updated = QtCore.Signal(str, float, object, object)
    rigid_body_updated = QtCore.Signal(str, object, object, object)
    rigid_component_selected = QtCore.Signal(str, object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Inspector", parent)
        self._entity_id: Optional[str] = None
        self._block_updates = False
        self._block_component_updates = False

        self._layout = QtWidgets.QFormLayout(self)
        self._id_field = QtWidgets.QLineEdit()
        self._id_field.setReadOnly(True)
        self._type_field = QtWidgets.QLineEdit()
        self._type_field.setReadOnly(True)

        self._mass_field = QtWidgets.QDoubleSpinBox()
        self._mass_field.setRange(1e-6, 1e9)
        self._mass_field.setDecimals(6)

        self._pos_x = self._make_spin()
        self._pos_y = self._make_spin()
        self._vel_x = self._make_spin()
        self._vel_y = self._make_spin()

        self._total_mass_field = QtWidgets.QLineEdit()
        self._total_mass_field.setReadOnly(True)

        self._com_x = self._make_spin()
        self._com_y = self._make_spin()
        self._com_z = self._make_spin()
        self._com_vel_x = self._make_spin()
        self._com_vel_y = self._make_spin()
        self._com_vel_z = self._make_spin()
        self._omega_x = self._make_spin()
        self._omega_y = self._make_spin()
        self._omega_z = self._make_spin()

        self._orientation_field = QtWidgets.QLineEdit()
        self._orientation_field.setReadOnly(True)

        self._components_table = QtWidgets.QTableWidget(0, 5)
        self._components_table.setHorizontalHeaderLabels(["ID", "Mass", "x", "y", "z"])
        self._components_table.horizontalHeader().setStretchLastSection(True)
        self._components_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._components_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._components_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self._layout.addRow("ID", self._id_field)
        self._layout.addRow("Type", self._type_field)
        self._layout.addRow("Mass", self._mass_field)
        self._layout.addRow("Pos X", self._pos_x)
        self._layout.addRow("Pos Y", self._pos_y)
        self._layout.addRow("Vel X", self._vel_x)
        self._layout.addRow("Vel Y", self._vel_y)

        self._layout.addRow("Total Mass", self._total_mass_field)
        self._layout.addRow("CoM X", self._com_x)
        self._layout.addRow("CoM Y", self._com_y)
        self._layout.addRow("CoM Z", self._com_z)
        self._layout.addRow("v_C X", self._com_vel_x)
        self._layout.addRow("v_C Y", self._com_vel_y)
        self._layout.addRow("v_C Z", self._com_vel_z)
        self._layout.addRow("omega X (body)", self._omega_x)
        self._layout.addRow("omega Y (body)", self._omega_y)
        self._layout.addRow("omega Z (body)", self._omega_z)
        self._layout.addRow("q (w,x,y,z)", self._orientation_field)
        self._layout.addRow("Components", self._components_table)

        self._mass_field.valueChanged.connect(self._emit_update)
        self._pos_x.valueChanged.connect(self._emit_update)
        self._pos_y.valueChanged.connect(self._emit_update)
        self._vel_x.valueChanged.connect(self._emit_update)
        self._vel_y.valueChanged.connect(self._emit_update)

        self._com_x.valueChanged.connect(self._emit_rigid_update)
        self._com_y.valueChanged.connect(self._emit_rigid_update)
        self._com_z.valueChanged.connect(self._emit_rigid_update)
        self._com_vel_x.valueChanged.connect(self._emit_rigid_update)
        self._com_vel_y.valueChanged.connect(self._emit_rigid_update)
        self._com_vel_z.valueChanged.connect(self._emit_rigid_update)
        self._omega_x.valueChanged.connect(self._emit_rigid_update)
        self._omega_y.valueChanged.connect(self._emit_rigid_update)
        self._omega_z.valueChanged.connect(self._emit_rigid_update)
        self._components_table.itemSelectionChanged.connect(self._emit_component_selected)

        self._point_fields = [
            self._mass_field,
            self._pos_x,
            self._pos_y,
            self._vel_x,
            self._vel_y,
        ]
        self._rigid_fields = [
            self._total_mass_field,
            self._com_x,
            self._com_y,
            self._com_z,
            self._com_vel_x,
            self._com_vel_y,
            self._com_vel_z,
            self._omega_x,
            self._omega_y,
            self._omega_z,
            self._orientation_field,
            self._components_table,
        ]

        self.setEnabled(False)

    def _make_spin(self) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(6)
        return spin

    def set_entity(self, entity: PointMass | RigidBody | None) -> None:
        self._block_updates = True
        self._block_component_updates = True
        if entity is None:
            self._entity_id = None
            self._id_field.setText("")
            self._type_field.setText("")
            self._components_table.setRowCount(0)
            self._components_table.clearSelection()
            self._set_group_visible(self._point_fields, False)
            self._set_group_visible(self._rigid_fields, False)
            self.setEnabled(False)
        else:
            self._entity_id = entity.entity_id
            self._id_field.setText(entity.entity_id)
            if isinstance(entity, PointMass):
                self._type_field.setText("PointMass")
                self._mass_field.setValue(entity.mass)
                self._pos_x.setValue(float(entity.position[0]))
                self._pos_y.setValue(float(entity.position[1]))
                self._vel_x.setValue(float(entity.velocity[0]))
                self._vel_y.setValue(float(entity.velocity[1]))
                self._components_table.setRowCount(0)
                self._components_table.clearSelection()
                self._set_group_visible(self._point_fields, True)
                self._set_group_visible(self._rigid_fields, False)
            else:
                self._type_field.setText("RigidBody")
                self._total_mass_field.setText(f"{entity.total_mass:.6f}")
                self._com_x.setValue(float(entity.com_position[0]))
                self._com_y.setValue(float(entity.com_position[1]))
                self._com_z.setValue(float(entity.com_position[2]))
                self._com_vel_x.setValue(float(entity.com_velocity[0]))
                self._com_vel_y.setValue(float(entity.com_velocity[1]))
                self._com_vel_z.setValue(float(entity.com_velocity[2]))
                self._omega_x.setValue(float(entity.omega_world[0]))
                self._omega_y.setValue(float(entity.omega_world[1]))
                self._omega_z.setValue(float(entity.omega_world[2]))
                self._orientation_field.setText(self._format_quaternion(entity.orientation))
                self._populate_components_table(entity)
                self._components_table.clearSelection()
                self._set_group_visible(self._point_fields, False)
                self._set_group_visible(self._rigid_fields, True)
            self.setEnabled(True)
        self._block_updates = False
        self._block_component_updates = False

    def _emit_update(self) -> None:
        if self._block_updates or self._entity_id is None:
            return
        self.point_mass_updated.emit(
            self._entity_id,
            float(self._mass_field.value()),
            (float(self._pos_x.value()), float(self._pos_y.value())),
            (float(self._vel_x.value()), float(self._vel_y.value())),
        )

    def _emit_rigid_update(self) -> None:
        if self._block_updates or self._entity_id is None:
            return
        self.rigid_body_updated.emit(
            self._entity_id,
            (
                float(self._com_x.value()),
                float(self._com_y.value()),
                float(self._com_z.value()),
            ),
            (
                float(self._com_vel_x.value()),
                float(self._com_vel_y.value()),
                float(self._com_vel_z.value()),
            ),
            (
                float(self._omega_x.value()),
                float(self._omega_y.value()),
                float(self._omega_z.value()),
            ),
        )

    def _populate_components_table(self, entity: RigidBody) -> None:
        self._components_table.setRowCount(len(entity.components))
        for row, component in enumerate(entity.components):
            self._components_table.setItem(row, 0, QtWidgets.QTableWidgetItem(component.component_id))
            self._components_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{component.mass:.6f}"))
            self._components_table.setItem(
                row,
                2,
                QtWidgets.QTableWidgetItem(f"{component.position_body[0]:.6f}"),
            )
            self._components_table.setItem(
                row,
                3,
                QtWidgets.QTableWidgetItem(f"{component.position_body[1]:.6f}"),
            )
            self._components_table.setItem(
                row,
                4,
                QtWidgets.QTableWidgetItem(f"{component.position_body[2]:.6f}"),
            )
        self._components_table.resizeColumnsToContents()

    def _set_group_visible(self, widgets: list[QtWidgets.QWidget], visible: bool) -> None:
        for widget in widgets:
            label = self._layout.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
            widget.setVisible(visible)

    @staticmethod
    def _format_quaternion(q: object) -> str:
        values = np.asarray(q, dtype=float).reshape(-1)
        if values.size != 4:
            return ""
        return f"[{values[0]:.6f}, {values[1]:.6f}, {values[2]:.6f}, {values[3]:.6f}]"

    def _emit_component_selected(self) -> None:
        if self._block_component_updates or self._entity_id is None:
            return
        selected = self._components_table.selectedItems()
        if not selected:
            self.rigid_component_selected.emit(self._entity_id, None)
            return
        row = selected[0].row()
        component_item = self._components_table.item(row, 0)
        if component_item is None:
            self.rigid_component_selected.emit(self._entity_id, None)
            return
        self.rigid_component_selected.emit(self._entity_id, component_item.text())
