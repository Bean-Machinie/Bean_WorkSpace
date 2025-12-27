from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.model import PointMass
from ..core.physics import center_of_mass
from ..core.scenarios import load_builtin_scenarios, scenario_registry
from ..core.sim import Simulation, SymplecticEulerIntegrator
from ..io.scene_format import SceneData, capture_scene, deserialize_scene, serialize_scene
from .widgets import InspectorPanel, InvariantsPanel, SceneView


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Space Dynamics Workbench")
        self.resize(1100, 700)

        load_builtin_scenarios()

        self._scene_view = SceneView()
        self._inspector = InspectorPanel()
        self._invariants = InvariantsPanel()

        side_layout = QtWidgets.QVBoxLayout()
        side_layout.addWidget(self._inspector)
        side_layout.addWidget(self._invariants)
        side_layout.addStretch(1)

        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side_layout)
        side_widget.setMinimumWidth(280)

        central_layout = QtWidgets.QHBoxLayout()
        central_layout.addWidget(self._scene_view, stretch=3)
        central_layout.addWidget(side_widget, stretch=1)

        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._play_action = QtGui.QAction("Play", self)
        self._play_action.setCheckable(True)
        self._play_action.triggered.connect(self._toggle_play)

        self._step_action = QtGui.QAction("Step", self)
        self._step_action.triggered.connect(self._single_step)

        self._reset_action = QtGui.QAction("Reset", self)
        self._reset_action.triggered.connect(self._reset_scene)

        self._save_action = QtGui.QAction("Save Scene", self)
        self._save_action.triggered.connect(self._save_scene)

        self._load_action = QtGui.QAction("Load Scene", self)
        self._load_action.triggered.connect(self._load_scene)

        self._scenario_combo = QtWidgets.QComboBox()
        for scenario in scenario_registry.all():
            self._scenario_combo.addItem(scenario.name, scenario.scenario_id)
        self._scenario_combo.currentIndexChanged.connect(self._scenario_changed)

        toolbar = self.addToolBar("Controls")
        toolbar.setMovable(False)
        toolbar.addAction(self._play_action)
        toolbar.addAction(self._step_action)
        toolbar.addAction(self._reset_action)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Scenario:"))
        toolbar.addWidget(self._scenario_combo)
        toolbar.addSeparator()
        toolbar.addAction(self._save_action)
        toolbar.addAction(self._load_action)

        self._scene_view.entity_selected.connect(self._on_entity_selected)
        self._scene_view.entity_dragged.connect(self._on_entity_dragged)
        self._inspector.entity_updated.connect(self._on_entity_updated)

        self._simulation: Simulation | None = None
        self._scenario_id: str | None = None
        self._initial_scene: SceneData | None = None
        self._selected_entity_id: Optional[str] = None

        if self._scenario_combo.count() > 0:
            self._scenario_combo.setCurrentIndex(0)
            self._scenario_changed(0)

    def _scenario_changed(self, index: int) -> None:
        scenario_id = self._scenario_combo.itemData(index)
        if scenario_id is None:
            return
        scenario = scenario_registry.get(scenario_id)
        sim = scenario.create_simulation()
        self._set_simulation(sim, scenario_id)
        defaults = scenario.ui_defaults()
        if defaults and defaults.view_range:
            self._scene_view.set_view_range(defaults.view_range)

    def _set_simulation(self, sim: Simulation, scenario_id: str | None) -> None:
        self._simulation = sim
        self._scenario_id = scenario_id
        self._selected_entity_id = None
        self._initial_scene = capture_scene(sim.entities, scenario_id=scenario_id)
        self._update_ui()

    def _update_ui(self) -> None:
        if self._simulation is None:
            return
        com = center_of_mass(self._simulation.entities)
        self._scene_view.update_scene(self._simulation.entities, com, self._selected_entity_id)
        self._invariants.update_values(self._simulation.entities)
        self._inspector.set_entity(self._find_entity(self._selected_entity_id))

    def _toggle_play(self, checked: bool) -> None:
        if self._simulation is None:
            return
        if checked:
            self._play_action.setText("Pause")
            interval_ms = int(self._simulation.dt * 1000)
            self._timer.start(max(interval_ms, 1))
        else:
            self._play_action.setText("Play")
            self._timer.stop()

    def _single_step(self) -> None:
        if self._simulation is None:
            return
        if self._timer.isActive():
            return
        self._simulation.step()
        self._update_ui()

    def _on_tick(self) -> None:
        if self._simulation is None:
            return
        self._simulation.step()
        self._update_ui()

    def _on_entity_selected(self, entity_id: str) -> None:
        self._selected_entity_id = entity_id
        self._update_ui()

    def _on_entity_dragged(self, entity_id: str, new_position: np.ndarray) -> None:
        entity = self._find_entity(entity_id)
        if entity is None:
            return
        entity.position = new_position
        self._update_ui()

    def _on_entity_updated(self, entity_id: str, mass: float, position: tuple[float, float], velocity: tuple[float, float]) -> None:
        entity = self._find_entity(entity_id)
        if entity is None:
            return
        entity.mass = mass
        entity.position = np.array(position, dtype=float)
        entity.velocity = np.array(velocity, dtype=float)
        self._update_ui()

    def _reset_scene(self) -> None:
        if self._initial_scene is None:
            return
        self._apply_scene(self._initial_scene)

    def _apply_scene(self, scene: SceneData) -> None:
        scenario_id = scene.scenario_id
        if scenario_id and scenario_id in {s.scenario_id for s in scenario_registry.all()}:
            scenario = scenario_registry.get(scenario_id)
            sim = scenario.create_simulation()
            sim.entities = scene.entities
        else:
            sim = Simulation(entities=scene.entities, dt=0.05, integrator=SymplecticEulerIntegrator())
        self._set_simulation(sim, scenario_id)
        self._select_scenario_in_combo(scenario_id)

    def _select_scenario_in_combo(self, scenario_id: str | None) -> None:
        self._scenario_combo.blockSignals(True)
        if scenario_id is None:
            self._scenario_combo.setCurrentIndex(-1)
        else:
            index = self._scenario_combo.findData(scenario_id)
            if index >= 0:
                self._scenario_combo.setCurrentIndex(index)
            else:
                self._scenario_combo.setCurrentIndex(-1)
        self._scenario_combo.blockSignals(False)

    def _save_scene(self) -> None:
        if self._simulation is None:
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Scene",
            "scene.json",
            "Scene Files (*.json)",
        )
        if not file_path:
            return
        scene = capture_scene(self._simulation.entities, scenario_id=self._scenario_id)
        payload = serialize_scene(scene)
        Path(file_path).write_text(payload, encoding="utf-8")

    def _load_scene(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Scene",
            "",
            "Scene Files (*.json)",
        )
        if not file_path:
            return
        payload = Path(file_path).read_text(encoding="utf-8")
        scene = deserialize_scene(payload)
        self._apply_scene(scene)
        self._initial_scene = scene

    def _find_entity(self, entity_id: str | None) -> PointMass | None:
        if self._simulation is None or entity_id is None:
            return None
        for entity in self._simulation.entities:
            if entity.entity_id == entity_id:
                return entity
        return None
