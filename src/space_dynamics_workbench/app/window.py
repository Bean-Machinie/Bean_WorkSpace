from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.model import PointMass, RigidBody, resolve_entity_by_id
from ..core.physics import FrameChoice, compute_frame_vectors, from_frame
from ..core.scenarios import load_builtin_scenarios, scenario_registry
from ..core.sim import Simulation, SymplecticEulerIntegrator
from ..io.scene_format import SceneData, capture_scene, clone_scene, deserialize_scene, serialize_scene
from .rendering import OverlayOptions, Renderer2D, Renderer3D, renderer3d_available, renderer3d_error
from .widgets import InspectorPanel, InvariantsPanel, ViewOptionsPanel


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Space Dynamics Workbench")
        self.resize(1100, 700)

        load_builtin_scenarios()

        self._renderer_2d = Renderer2D()
        self._renderer_3d: Renderer2D | Renderer3D | None = None
        self._renderer = self._renderer_2d
        self._inspector = InspectorPanel()
        self._invariants = InvariantsPanel()
        self._view_options = ViewOptionsPanel()
        self._view_options.set_renderer_availability(
            renderer3d_available(),
            renderer3d_error() or "Install 3D extras (PyOpenGL) to enable 3D mode.",
        )

        side_layout = QtWidgets.QVBoxLayout()
        side_layout.addWidget(self._view_options)
        side_layout.addWidget(self._inspector)
        side_layout.addWidget(self._invariants)
        side_layout.addStretch(1)

        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side_layout)
        side_widget.setMinimumWidth(280)

        self._renderer_stack = QtWidgets.QStackedWidget()
        self._renderer_stack.addWidget(self._renderer_2d)

        central_layout = QtWidgets.QHBoxLayout()
        central_layout.addWidget(self._renderer_stack, stretch=3)
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

        self._connect_renderer(self._renderer)
        self._inspector.point_mass_updated.connect(self._on_point_mass_updated)
        self._inspector.rigid_body_updated.connect(self._on_rigid_body_updated)
        self._view_options.frame_changed.connect(self._on_frame_changed)
        self._view_options.overlays_changed.connect(self._on_overlays_changed)
        self._view_options.renderer_changed.connect(self._on_renderer_changed)

        self._simulation: Simulation | None = None
        self._scenario_id: str | None = None
        self._initial_scene: SceneData | None = None
        self._selected_entity_id: Optional[str] = None
        self._frame_choice: FrameChoice = FrameChoice.WORLD
        self._overlays = OverlayOptions()
        self._frame_vectors = None

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
            self._renderer.set_view_range(defaults.view_range)

    def _set_simulation(self, sim: Simulation, scenario_id: str | None) -> None:
        self._simulation = sim
        self._scenario_id = scenario_id
        self._selected_entity_id = None
        self._initial_scene = capture_scene(sim.entities, scenario_id=scenario_id)
        self._update_ui()

    def _update_ui(self) -> None:
        if self._simulation is None:
            return
        self._frame_vectors = compute_frame_vectors(self._simulation.entities, self._frame_choice)
        self._renderer.set_scene(self._frame_vectors, self._overlays, self._selected_entity_id)
        self._invariants.update_values(self._simulation.entities)
        entity, _ = self._resolve_entity(self._selected_entity_id)
        self._inspector.set_entity(entity)

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
        entity, component = self._resolve_entity(entity_id)
        if entity is None or component is not None or self._frame_vectors is None:
            return
        if isinstance(entity, RigidBody):
            return
        dimension = self._frame_vectors.dimension
        position_frame = np.zeros(dimension, dtype=float)
        position_frame[:2] = new_position[:2]
        entity.position = from_frame(position_frame, self._frame_choice, self._frame_vectors.r_oc_world)
        self._update_ui()

    def _on_point_mass_updated(
        self,
        entity_id: str,
        mass: float,
        position: tuple[float, float],
        velocity: tuple[float, float],
    ) -> None:
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, PointMass):
            return
        entity.mass = mass
        position_frame = np.array(position, dtype=float)
        if self._frame_vectors is not None:
            position_vec = np.zeros(self._frame_vectors.dimension, dtype=float)
            position_vec[:2] = position_frame[:2]
            position_world = from_frame(position_vec, self._frame_choice, self._frame_vectors.r_oc_world)
        else:
            position_world = position_frame
        entity.position = position_world
        entity.velocity = np.array(velocity, dtype=float)
        self._update_ui()

    def _on_rigid_body_updated(
        self,
        entity_id: str,
        com_position: tuple[float, float, float],
        com_velocity: tuple[float, float, float],
        omega_world: tuple[float, float, float],
    ) -> None:
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            return
        entity.com_position = np.array(com_position, dtype=float)
        entity.com_velocity = np.array(com_velocity, dtype=float)
        entity.omega_world = np.array(omega_world, dtype=float)
        self._update_ui()

    def _reset_scene(self) -> None:
        if self._initial_scene is None:
            return
        self._apply_scene(clone_scene(self._initial_scene))

    def _apply_scene(self, scene: SceneData) -> None:
        scene_copy = clone_scene(scene)
        scenario_id = scene_copy.scenario_id
        if scenario_id and scenario_id in {s.scenario_id for s in scenario_registry.all()}:
            scenario = scenario_registry.get(scenario_id)
            sim = scenario.create_simulation()
            sim.entities = scene_copy.entities
        else:
            sim = Simulation(entities=scene_copy.entities, dt=0.05, integrator=SymplecticEulerIntegrator())
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

    def _on_frame_changed(self, frame: FrameChoice) -> None:
        self._frame_choice = frame
        self._update_ui()

    def _on_overlays_changed(self, overlays: OverlayOptions) -> None:
        self._overlays = overlays
        self._update_ui()

    def _on_renderer_changed(self, mode: str) -> None:
        if mode == "3d":
            if not renderer3d_available():
                return
            renderer = self._ensure_renderer3d()
        else:
            renderer = self._renderer_2d
        self._swap_renderer(renderer)

    def _ensure_renderer3d(self) -> Renderer2D | Renderer3D:
        if self._renderer_3d is None:
            if Renderer3D is None:
                return self._renderer_2d
            self._renderer_3d = Renderer3D()
            self._renderer_stack.addWidget(self._renderer_3d)
        return self._renderer_3d

    def _swap_renderer(self, renderer) -> None:
        if renderer is self._renderer:
            return
        self._disconnect_renderer(self._renderer)
        self._renderer = renderer
        self._connect_renderer(renderer)
        self._renderer_stack.setCurrentWidget(renderer)
        self._update_ui()

    def _connect_renderer(self, renderer) -> None:
        renderer.entity_selected.connect(self._on_entity_selected)
        renderer.entity_dragged.connect(self._on_entity_dragged)

    def _disconnect_renderer(self, renderer) -> None:
        try:
            renderer.entity_selected.disconnect(self._on_entity_selected)
        except (TypeError, RuntimeError):
            pass
        try:
            renderer.entity_dragged.disconnect(self._on_entity_dragged)
        except (TypeError, RuntimeError):
            pass

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

    def _resolve_entity(
        self, entity_id: str | None
    ) -> tuple[PointMass | RigidBody | None, object | None]:
        if self._simulation is None or entity_id is None:
            return None, None
        return resolve_entity_by_id(self._simulation.entities, entity_id)
