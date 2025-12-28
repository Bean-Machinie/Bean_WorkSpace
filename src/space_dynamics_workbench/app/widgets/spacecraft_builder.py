from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from ...core.model import RigidBody, RigidBodyComponent
from ..rendering.base import DisplayOptions


@dataclass
class MeshInfo:
    path: str = ""
    info: str = "-"
    warning: str = ""


class SpacecraftEditorPanel(QtWidgets.QWidget):
    mesh_load_requested = QtCore.Signal()
    display_options_changed = QtCore.Signal(object)
    components_changed = QtCore.Signal(str, object)
    component_selected = QtCore.Signal(str, object)
    recenter_requested = QtCore.Signal(str)
    initial_state_changed = QtCore.Signal(str, object, object, object, object)
    reset_spacecraft_requested = QtCore.Signal(str)
    auto_generate_requested = QtCore.Signal(str, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._entity_id: Optional[str] = None
        self._block_updates = False
        self._block_selection = False

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer_layout.addWidget(scroll_area)

        content = QtWidgets.QWidget()
        scroll_area.setWidget(content)
        main_layout = QtWidgets.QVBoxLayout(content)
        main_layout.setContentsMargins(8, 8, 8, 8)

        model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        button_row = QtWidgets.QHBoxLayout()
        self._load_button = QtWidgets.QPushButton("Load Model...")
        self._load_button.clicked.connect(self.mesh_load_requested.emit)
        self._new_craft_button = QtWidgets.QPushButton("Reset Spacecraft")
        self._new_craft_button.clicked.connect(self._emit_reset_spacecraft)
        button_row.addWidget(self._load_button)
        button_row.addWidget(self._new_craft_button)
        button_row.addStretch(1)
        model_layout.addLayout(button_row)

        self._model_path = QtWidgets.QLineEdit()
        self._model_path.setReadOnly(True)
        self._model_info = QtWidgets.QLabel("-")
        self._model_warning = QtWidgets.QLabel("")
        self._model_warning.setStyleSheet("color: #d84315;")
        model_layout.addWidget(self._model_path)
        model_layout.addWidget(self._model_info)
        model_layout.addWidget(self._model_warning)

        display_group = QtWidgets.QGroupBox("Display")
        display_layout = QtWidgets.QFormLayout(display_group)
        self._show_mesh = QtWidgets.QCheckBox("Show Mesh")
        self._show_mesh.setChecked(True)
        self._show_mass_points = QtWidgets.QCheckBox("Show Mass Points")
        self._show_mass_points.setChecked(True)
        self._opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(35)
        self._opacity_label = QtWidgets.QLabel("0.35")
        display_layout.addRow(self._show_mesh)
        display_layout.addRow(self._show_mass_points)
        display_layout.addRow("Mesh Opacity", self._opacity_slider)
        display_layout.addRow("", self._opacity_label)

        self._show_mesh.toggled.connect(self._emit_display_options)
        self._show_mass_points.toggled.connect(self._emit_display_options)
        self._opacity_slider.valueChanged.connect(self._emit_display_options)

        components_group = QtWidgets.QGroupBox("Mass Components")
        components_layout = QtWidgets.QVBoxLayout(components_group)
        self._components_table = QtWidgets.QTableWidget(0, 5)
        self._components_table.setHorizontalHeaderLabels(["ID", "Mass", "x", "y", "z"])
        self._components_table.horizontalHeader().setStretchLastSection(True)
        self._components_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._components_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._components_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.SelectedClicked
        )
        self._components_table.itemChanged.connect(self._emit_components_changed)
        self._components_table.itemSelectionChanged.connect(self._emit_component_selected)

        buttons_row = QtWidgets.QHBoxLayout()
        self._add_button = QtWidgets.QPushButton("Add")
        self._remove_button = QtWidgets.QPushButton("Remove")
        self._duplicate_button = QtWidgets.QPushButton("Duplicate")
        self._auto_generate_button = QtWidgets.QPushButton("Auto-generate mass points")
        self._auto_generate_count = QtWidgets.QSpinBox()
        self._auto_generate_count.setRange(1, 50)
        self._auto_generate_count.setValue(5)
        self._add_button.clicked.connect(self._add_component)
        self._remove_button.clicked.connect(self._remove_component)
        self._duplicate_button.clicked.connect(self._duplicate_component)
        self._auto_generate_button.clicked.connect(self._emit_auto_generate)
        buttons_row.addWidget(self._add_button)
        buttons_row.addWidget(self._remove_button)
        buttons_row.addWidget(self._duplicate_button)
        buttons_row.addWidget(self._auto_generate_button)
        buttons_row.addWidget(QtWidgets.QLabel("Max points"))
        buttons_row.addWidget(self._auto_generate_count)
        buttons_row.addStretch(1)

        components_layout.addWidget(self._components_table)
        com_note = QtWidgets.QLabel(
            "Note: components auto-recenter so the body origin matches CoM; mesh offset updates to stay aligned."
        )
        com_note.setWordWrap(True)
        components_layout.addWidget(com_note)
        components_layout.addLayout(buttons_row)

        com_group = QtWidgets.QGroupBox("Center of Mass")
        com_layout = QtWidgets.QFormLayout(com_group)
        self._com_body_label = QtWidgets.QLabel("-")
        self._com_world_label = QtWidgets.QLabel("-")
        self._recenter_button = QtWidgets.QPushButton("Recenter components to CoM")
        self._recenter_button.clicked.connect(self._emit_recenter_requested)
        com_layout.addRow("Body frame (r_C)", self._com_body_label)
        com_layout.addRow("World frame", self._com_world_label)
        com_layout.addRow(self._recenter_button)

        state_group = QtWidgets.QGroupBox("Initial State (World Frame)")
        state_layout = QtWidgets.QFormLayout(state_group)
        self._state_pos_x = self._make_spin()
        self._state_pos_y = self._make_spin()
        self._state_pos_z = self._make_spin()
        self._state_vel_x = self._make_spin()
        self._state_vel_y = self._make_spin()
        self._state_vel_z = self._make_spin()
        self._state_q_w = self._make_spin()
        self._state_q_x = self._make_spin()
        self._state_q_y = self._make_spin()
        self._state_q_z = self._make_spin()
        self._state_omega_x = self._make_spin()
        self._state_omega_y = self._make_spin()
        self._state_omega_z = self._make_spin()
        self._state_reset_button = QtWidgets.QPushButton("Reset to defaults")
        self._state_reset_button.clicked.connect(self._reset_state_defaults)

        state_layout.addRow("CoM X", self._state_pos_x)
        state_layout.addRow("CoM Y", self._state_pos_y)
        state_layout.addRow("CoM Z", self._state_pos_z)
        state_layout.addRow("v_C X", self._state_vel_x)
        state_layout.addRow("v_C Y", self._state_vel_y)
        state_layout.addRow("v_C Z", self._state_vel_z)
        state_layout.addRow("q w", self._state_q_w)
        state_layout.addRow("q x", self._state_q_x)
        state_layout.addRow("q y", self._state_q_y)
        state_layout.addRow("q z", self._state_q_z)
        state_layout.addRow("omega X", self._state_omega_x)
        state_layout.addRow("omega Y", self._state_omega_y)
        state_layout.addRow("omega Z", self._state_omega_z)
        state_layout.addRow(self._state_reset_button)

        main_layout.addWidget(model_group)
        main_layout.addWidget(display_group)
        main_layout.addWidget(components_group, stretch=1)
        main_layout.addWidget(com_group)
        main_layout.addWidget(state_group)
        main_layout.addStretch(1)

        self.setEnabled(False)

        self._state_fields = [
            self._state_pos_x,
            self._state_pos_y,
            self._state_pos_z,
            self._state_vel_x,
            self._state_vel_y,
            self._state_vel_z,
            self._state_q_w,
            self._state_q_x,
            self._state_q_y,
            self._state_q_z,
            self._state_omega_x,
            self._state_omega_y,
            self._state_omega_z,
        ]
        for field in self._state_fields:
            field.valueChanged.connect(self._emit_state_updated)

    def set_mesh_loading_available(self, available: bool, message: str | None = None) -> None:
        self._load_button.setEnabled(available)
        self._auto_generate_button.setEnabled(available)
        self._auto_generate_count.setEnabled(available)
        if not available:
            self._load_button.setToolTip(message or "Install mesh extras to enable loading.")
            self._auto_generate_button.setToolTip(message or "Install mesh extras to enable auto generation.")
            self._auto_generate_count.setToolTip(message or "Install mesh extras to enable auto generation.")
        else:
            self._load_button.setToolTip("")
            self._auto_generate_button.setToolTip("")
            self._auto_generate_count.setToolTip("")

    def set_entity(self, entity: RigidBody | None) -> None:
        self._block_updates = True
        self._block_selection = True
        if entity is None:
            self._entity_id = None
            self._components_table.setRowCount(0)
            self._model_path.setText("")
            self._model_info.setText("-")
            self._model_warning.setText("")
            self._com_body_label.setText("-")
            self._com_world_label.setText("-")
            self._set_state_defaults()
            self.setEnabled(False)
        else:
            self._entity_id = entity.entity_id
            self._populate_components_table(entity)
            if entity.mesh is not None:
                if self._model_path.text() != entity.mesh.path:
                    self._model_info.setText("-")
            else:
                self._model_info.setText("-")
            if entity.mesh is not None:
                self._model_path.setText(entity.mesh.path)
                if entity.mesh.path_is_absolute:
                    self._model_warning.setText("External path (absolute) reduces portability.")
                else:
                    self._model_warning.setText("")
            else:
                self._model_path.setText("")
                self._model_warning.setText("")
            self._update_com_labels(entity)
            self._populate_state_fields(entity)
            self.setEnabled(True)
        self._block_updates = False
        self._block_selection = False

    def set_mesh_info(self, info: MeshInfo) -> None:
        self._model_path.setText(info.path)
        self._model_info.setText(info.info)
        self._model_warning.setText(info.warning)

    def set_selected_component(self, component_id: str | None) -> None:
        if self._block_selection:
            return
        if component_id is None:
            self._components_table.clearSelection()
            return
        for row in range(self._components_table.rowCount()):
            item = self._components_table.item(row, 0)
            if item and item.text() == component_id:
                self._components_table.selectRow(row)
                return

    def display_options(self) -> DisplayOptions:
        return DisplayOptions(
            show_mesh=self._show_mesh.isChecked(),
            show_mass_points=self._show_mass_points.isChecked(),
            mesh_opacity=float(self._opacity_slider.value() / 100.0),
        )

    def _emit_display_options(self) -> None:
        value = self._opacity_slider.value() / 100.0
        self._opacity_label.setText(f"{value:.2f}")
        self.display_options_changed.emit(self.display_options())

    def _emit_components_changed(self, _: QtWidgets.QTableWidgetItem | None) -> None:
        if self._block_updates or self._entity_id is None:
            return
        components = self._collect_components()
        if not components:
            return
        self.components_changed.emit(self._entity_id, components)

    def _emit_component_selected(self) -> None:
        if self._block_selection or self._entity_id is None:
            return
        selected = self._components_table.selectedItems()
        if not selected:
            self.component_selected.emit(self._entity_id, None)
            return
        row = selected[0].row()
        component_item = self._components_table.item(row, 0)
        if component_item is None:
            self.component_selected.emit(self._entity_id, None)
            return
        self.component_selected.emit(self._entity_id, component_item.text())

    def _emit_recenter_requested(self) -> None:
        if self._entity_id is None:
            return
        self.recenter_requested.emit(self._entity_id)

    def _emit_state_updated(self) -> None:
        if self._block_updates or self._entity_id is None:
            return
        self.initial_state_changed.emit(
            self._entity_id,
            (
                float(self._state_pos_x.value()),
                float(self._state_pos_y.value()),
                float(self._state_pos_z.value()),
            ),
            (
                float(self._state_vel_x.value()),
                float(self._state_vel_y.value()),
                float(self._state_vel_z.value()),
            ),
            (
                float(self._state_q_w.value()),
                float(self._state_q_x.value()),
                float(self._state_q_y.value()),
                float(self._state_q_z.value()),
            ),
            (
                float(self._state_omega_x.value()),
                float(self._state_omega_y.value()),
                float(self._state_omega_z.value()),
            ),
        )

    def _emit_reset_spacecraft(self) -> None:
        if self._entity_id is None:
            return
        self.reset_spacecraft_requested.emit(self._entity_id)

    def _emit_auto_generate(self) -> None:
        if self._entity_id is None:
            return
        self.auto_generate_requested.emit(self._entity_id, int(self._auto_generate_count.value()))

    def _populate_components_table(self, entity: RigidBody) -> None:
        self._components_table.setRowCount(len(entity.components))
        for row, component in enumerate(entity.components):
            self._set_item(row, 0, component.component_id)
            self._set_item(row, 1, f"{component.mass:.6f}")
            self._set_item(row, 2, f"{component.position_body[0]:.6f}")
            self._set_item(row, 3, f"{component.position_body[1]:.6f}")
            self._set_item(row, 4, f"{component.position_body[2]:.6f}")
        self._components_table.resizeColumnsToContents()

    def _set_item(self, row: int, col: int, value: str) -> None:
        item = self._components_table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            self._components_table.setItem(row, col, item)
        item.setText(value)

    def _collect_components(self) -> list[RigidBodyComponent]:
        components: list[RigidBodyComponent] = []
        used_ids: set[str] = set()
        for row in range(self._components_table.rowCount()):
            comp_id = self._text_or_default(row, 0, f"C{row + 1}")
            comp_id = self._dedupe_id(comp_id, used_ids)
            used_ids.add(comp_id)
            mass = max(self._float_or_default(row, 1, 1.0), 1e-6)
            x = self._float_or_default(row, 2, 0.0)
            y = self._float_or_default(row, 3, 0.0)
            z = self._float_or_default(row, 4, 0.0)
            components.append(
                RigidBodyComponent(
                    component_id=comp_id,
                    mass=float(mass),
                    position_body=np.array([x, y, z], dtype=float),
                )
            )
        return components

    def _text_or_default(self, row: int, col: int, default: str) -> str:
        item = self._components_table.item(row, col)
        if item is None:
            return default
        text = item.text().strip()
        return text if text else default

    def _float_or_default(self, row: int, col: int, default: float) -> float:
        item = self._components_table.item(row, col)
        if item is None:
            return default
        try:
            return float(item.text())
        except ValueError:
            return default

    @staticmethod
    def _dedupe_id(component_id: str, used: set[str]) -> str:
        if component_id not in used:
            return component_id
        suffix = 1
        while f"{component_id}_{suffix}" in used:
            suffix += 1
        return f"{component_id}_{suffix}"

    def _add_component(self) -> None:
        if self._entity_id is None:
            return
        self._block_updates = True
        row = self._components_table.rowCount()
        self._components_table.insertRow(row)
        self._set_item(row, 0, f"C{row + 1}")
        self._set_item(row, 1, "1.000000")
        self._set_item(row, 2, "0.000000")
        self._set_item(row, 3, "0.000000")
        self._set_item(row, 4, "0.000000")
        self._block_updates = False
        self._emit_components_changed(self._components_table.item(row, 0))

    def _remove_component(self) -> None:
        if self._entity_id is None:
            return
        if self._components_table.rowCount() <= 1:
            return
        row = self._components_table.currentRow()
        if row < 0:
            row = self._components_table.rowCount() - 1
        self._components_table.removeRow(row)
        self._emit_components_changed(self._components_table.item(max(row - 1, 0), 0))

    def _duplicate_component(self) -> None:
        if self._entity_id is None:
            return
        row = self._components_table.currentRow()
        if row < 0:
            row = self._components_table.rowCount() - 1
        if row < 0:
            return
        comp_id = self._text_or_default(row, 0, f"C{row + 1}")
        mass = self._float_or_default(row, 1, 1.0)
        x = self._float_or_default(row, 2, 0.0)
        y = self._float_or_default(row, 3, 0.0)
        z = self._float_or_default(row, 4, 0.0)
        new_row = self._components_table.rowCount()
        self._components_table.insertRow(new_row)
        self._set_item(new_row, 0, f"{comp_id}_copy")
        self._set_item(new_row, 1, f"{mass:.6f}")
        self._set_item(new_row, 2, f"{x:.6f}")
        self._set_item(new_row, 3, f"{y:.6f}")
        self._set_item(new_row, 4, f"{z:.6f}")
        self._emit_components_changed(self._components_table.item(new_row, 0))

    def _update_com_labels(self, entity: RigidBody) -> None:
        com_body = self._compute_com_body(entity)
        self._com_body_label.setText(self._format_vector(com_body))
        self._com_world_label.setText(self._format_vector(entity.com_position))

    @staticmethod
    def _compute_com_body(entity: RigidBody) -> np.ndarray:
        masses = np.array([component.mass for component in entity.components], dtype=float)
        positions = np.stack([component.position_body for component in entity.components])
        return np.sum(positions * masses[:, None], axis=0) / np.sum(masses)

    @staticmethod
    def _format_vector(vec: np.ndarray) -> str:
        return f"[{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}]"

    @staticmethod
    def _make_spin() -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(6)
        return spin

    def _populate_state_fields(self, entity: RigidBody) -> None:
        self._state_pos_x.setValue(float(entity.com_position[0]))
        self._state_pos_y.setValue(float(entity.com_position[1]))
        self._state_pos_z.setValue(float(entity.com_position[2]))
        self._state_vel_x.setValue(float(entity.com_velocity[0]))
        self._state_vel_y.setValue(float(entity.com_velocity[1]))
        self._state_vel_z.setValue(float(entity.com_velocity[2]))
        self._state_q_w.setValue(float(entity.orientation[0]))
        self._state_q_x.setValue(float(entity.orientation[1]))
        self._state_q_y.setValue(float(entity.orientation[2]))
        self._state_q_z.setValue(float(entity.orientation[3]))
        self._state_omega_x.setValue(float(entity.omega_world[0]))
        self._state_omega_y.setValue(float(entity.omega_world[1]))
        self._state_omega_z.setValue(float(entity.omega_world[2]))

    def _set_state_defaults(self) -> None:
        self._state_pos_x.setValue(0.0)
        self._state_pos_y.setValue(0.0)
        self._state_pos_z.setValue(0.0)
        self._state_vel_x.setValue(0.0)
        self._state_vel_y.setValue(0.0)
        self._state_vel_z.setValue(0.0)
        self._state_q_w.setValue(1.0)
        self._state_q_x.setValue(0.0)
        self._state_q_y.setValue(0.0)
        self._state_q_z.setValue(0.0)
        self._state_omega_x.setValue(0.0)
        self._state_omega_y.setValue(0.0)
        self._state_omega_z.setValue(0.0)

    def _reset_state_defaults(self) -> None:
        if self._block_updates:
            return
        self._set_state_defaults()
        self._emit_state_updated()
