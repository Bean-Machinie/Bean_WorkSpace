from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class ScenarioPanel(QtWidgets.QGroupBox):
    name_changed = QtCore.Signal(str)
    add_spacecraft_requested = QtCore.Signal()
    object_selected = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Scenario", parent)
        layout = QtWidgets.QFormLayout(self)

        self._name_edit = QtWidgets.QLineEdit()
        self._path_label = QtWidgets.QLabel("-")
        self._path_label.setWordWrap(True)
        self._add_spacecraft = QtWidgets.QPushButton("Add Spacecraft")
        self._add_spacecraft.clicked.connect(self.add_spacecraft_requested.emit)
        self._objects_list = QtWidgets.QListWidget()
        self._objects_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._objects_list.currentItemChanged.connect(self._emit_object_selected)

        layout.addRow("Name", self._name_edit)
        layout.addRow("File", self._path_label)
        layout.addRow("Objects", self._objects_list)
        layout.addRow(self._add_spacecraft)

        self._name_edit.editingFinished.connect(self._emit_name_changed)

    def set_scenario(self, name: str, path: str | None) -> None:
        self._name_edit.setText(name)
        self._path_label.setText(path or "-")

    def set_objects(self, objects: list[tuple[str, str]]) -> None:
        self._objects_list.blockSignals(True)
        self._objects_list.clear()
        for entity_id, label in objects:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, entity_id)
            self._objects_list.addItem(item)
        self._objects_list.blockSignals(False)

    def set_selected_object(self, entity_id: str | None) -> None:
        self._objects_list.blockSignals(True)
        if entity_id is None:
            self._objects_list.clearSelection()
            self._objects_list.blockSignals(False)
            return
        for row in range(self._objects_list.count()):
            item = self._objects_list.item(row)
            if item.data(QtCore.Qt.UserRole) == entity_id:
                self._objects_list.setCurrentItem(item)
                break
        self._objects_list.blockSignals(False)

    def _emit_name_changed(self) -> None:
        self.name_changed.emit(self._name_edit.text())

    def _emit_object_selected(self, current: QtWidgets.QListWidgetItem | None) -> None:
        if current is None:
            return
        entity_id = current.data(QtCore.Qt.UserRole)
        if entity_id:
            self.object_selected.emit(str(entity_id))
