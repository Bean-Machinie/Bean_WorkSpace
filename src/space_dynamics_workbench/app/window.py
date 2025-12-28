from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.model import MeshMetadata, PointMass, RigidBody, RigidBodyComponent, resolve_entity_by_id
from ..core.physics import FrameChoice, compute_frame_vectors, from_frame
from ..core.scenarios import load_builtin_scenarios, scenario_registry
from ..core.sim import Simulation, SymplecticEulerIntegrator
from ..io.scene_format import SceneData, capture_scene, clone_scene, deserialize_scene, serialize_scene
from .scenario_controller import ScenarioController
from .mesh_generation import generate_mass_points_from_mesh
from .mesh_loading import load_mesh_data, mesh_loading_available, mesh_loading_error
from .rendering import DisplayOptions, OverlayOptions, Renderer2D, Renderer3D, renderer3d_available, renderer3d_error
from .widgets import (
    InspectorPanel,
    InvariantsPanel,
    NewScenarioDialog,
    ScenarioSettingsDialog,
    SimulationPanel,
    SpacecraftEditorPanel,
)
from .widgets.spacecraft_builder import MeshInfo


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Space Dynamics Workbench")
        self.resize(1100, 700)

        load_builtin_scenarios()

        self._renderer_2d = Renderer2D()
        self._renderer_3d: Renderer2D | Renderer3D | None = None
        self._renderer = self._renderer_2d

        self._renderer_stack = QtWidgets.QStackedWidget()
        self._renderer_stack.addWidget(self._renderer_2d)
        self.setCentralWidget(self._renderer_stack)
        self.setDockOptions(
            QtWidgets.QMainWindow.AllowTabbedDocks
            | QtWidgets.QMainWindow.AllowNestedDocks
            | QtWidgets.QMainWindow.AnimatedDocks
        )

        self._builder_panel = SpacecraftEditorPanel()
        self._builder_dock = QtWidgets.QDockWidget("Spacecraft Editor", self)
        self._builder_dock.setWidget(self._builder_panel)
        self._builder_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self._builder_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self._builder_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._builder_dock)

        self._simulation_panel = SimulationPanel()
        self._simulation_dock = QtWidgets.QDockWidget("Simulation", self)
        self._simulation_dock.setWidget(self._simulation_panel)
        self._simulation_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self._simulation_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self._simulation_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._simulation_dock)
        self._simulation_dock.hide()

        self._inspector = InspectorPanel()
        self._inspector.setTitle("")
        self._inspector_dock = QtWidgets.QDockWidget("Inspector", self)
        self._inspector_dock.setWidget(self._inspector)
        self._inspector_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self._inspector_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self._inspector.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._inspector_dock)

        self._invariants = InvariantsPanel()
        self._invariants.setTitle("")
        self._invariants_dock = QtWidgets.QDockWidget("Invariants / Truth Meter", self)
        self._invariants_dock.setWidget(self._invariants)
        self._invariants_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self._invariants_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        self._invariants.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._invariants_dock)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._scenario_controller = ScenarioController()

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

        self._frame_action = QtGui.QAction("Frame Scene", self)
        self._frame_action.triggered.connect(self._frame_scene)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        self._new_scenario_action = QtGui.QAction("New Scenario...", self)
        self._new_scenario_action.triggered.connect(self._new_scenario)
        self._open_scenario_action = QtGui.QAction("Open Scenario...", self)
        self._open_scenario_action.triggered.connect(self._open_scenario)
        self._save_scenario_action = QtGui.QAction("Save Scenario", self)
        self._save_scenario_action.triggered.connect(self._save_scenario)
        self._save_scenario_as_action = QtGui.QAction("Save Scenario As...", self)
        self._save_scenario_as_action.triggered.connect(self._save_scenario_as)

        file_menu.addAction(self._new_scenario_action)
        file_menu.addAction(self._open_scenario_action)
        file_menu.addSeparator()
        file_menu.addAction(self._save_scenario_action)
        file_menu.addAction(self._save_scenario_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self._save_action)
        file_menu.addAction(self._load_action)

        scenario_menu = menu_bar.addMenu("Scenario")
        self._builtin_scenario_menu = scenario_menu.addMenu("Built-in Scenarios")
        self._scenario_group = QtGui.QActionGroup(self)
        self._scenario_group.setExclusive(True)
        self._scenario_actions: dict[str, QtGui.QAction] = {}
        for scenario in scenario_registry.all():
            action = QtGui.QAction(scenario.name, self)
            action.setCheckable(True)
            action.setData(scenario.scenario_id)
            action.triggered.connect(self._on_scenario_action)
            self._scenario_group.addAction(action)
            self._builtin_scenario_menu.addAction(action)
            self._scenario_actions[scenario.scenario_id] = action

        edit_menu = menu_bar.addMenu("Edit")
        self._add_spacecraft_action = QtGui.QAction("Add Spacecraft...", self)
        self._add_spacecraft_action.triggered.connect(self._on_add_spacecraft_requested)
        self._scenario_settings_action = QtGui.QAction("Scenario Settings...", self)
        self._scenario_settings_action.triggered.connect(self._open_scenario_settings)

        edit_menu.addAction(self._add_spacecraft_action)
        edit_menu.addAction(self._scenario_settings_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self._play_action)
        edit_menu.addAction(self._step_action)
        edit_menu.addAction(self._reset_action)

        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(self._frame_action)
        view_menu.addSeparator()

        self._frame_menu = view_menu.addMenu("Reference Frame")
        self._frame_group = QtGui.QActionGroup(self)
        self._frame_group.setExclusive(True)
        self._frame_world_action = QtGui.QAction("World (O)", self)
        self._frame_world_action.setCheckable(True)
        self._frame_world_action.setData(FrameChoice.WORLD)
        self._frame_world_action.triggered.connect(
            lambda: self._on_frame_action(FrameChoice.WORLD)
        )
        self._frame_com_action = QtGui.QAction("Center of Mass (C)", self)
        self._frame_com_action.setCheckable(True)
        self._frame_com_action.setData(FrameChoice.COM)
        self._frame_com_action.triggered.connect(lambda: self._on_frame_action(FrameChoice.COM))
        self._frame_group.addAction(self._frame_world_action)
        self._frame_group.addAction(self._frame_com_action)
        self._frame_menu.addAction(self._frame_world_action)
        self._frame_menu.addAction(self._frame_com_action)

        self._renderer_menu = view_menu.addMenu("Renderer")
        self._renderer_group = QtGui.QActionGroup(self)
        self._renderer_group.setExclusive(True)
        self._renderer_2d_action = QtGui.QAction("2D (stable)", self)
        self._renderer_2d_action.setCheckable(True)
        self._renderer_2d_action.triggered.connect(lambda: self._on_renderer_changed("2d"))
        self._renderer_3d_action = QtGui.QAction("3D (experimental)", self)
        self._renderer_3d_action.setCheckable(True)
        self._renderer_3d_action.triggered.connect(lambda: self._on_renderer_changed("3d"))
        self._renderer_group.addAction(self._renderer_2d_action)
        self._renderer_group.addAction(self._renderer_3d_action)
        self._renderer_menu.addAction(self._renderer_2d_action)
        self._renderer_menu.addAction(self._renderer_3d_action)

        view_menu.addSeparator()
        self._show_axes_action = QtGui.QAction("Show Axes", self)
        self._show_axes_action.setCheckable(True)
        self._show_axes_action.toggled.connect(self._on_overlay_action)
        view_menu.addAction(self._show_axes_action)

        self._show_axis_labels_action = QtGui.QAction("Show Axis Labels", self)
        self._show_axis_labels_action.setCheckable(True)
        self._show_axis_labels_action.toggled.connect(self._on_overlay_action)
        view_menu.addAction(self._show_axis_labels_action)

        self._show_r_op_action = QtGui.QAction("Show r_OP", self)
        self._show_r_op_action.setCheckable(True)
        self._show_r_op_action.toggled.connect(self._on_overlay_action)
        view_menu.addAction(self._show_r_op_action)

        self._show_r_oc_action = QtGui.QAction("Show r_OC", self)
        self._show_r_oc_action.setCheckable(True)
        self._show_r_oc_action.toggled.connect(self._on_overlay_action)
        view_menu.addAction(self._show_r_oc_action)

        self._show_r_cp_action = QtGui.QAction("Show r_CP", self)
        self._show_r_cp_action.setCheckable(True)
        self._show_r_cp_action.toggled.connect(self._on_overlay_action)
        view_menu.addAction(self._show_r_cp_action)

        view_menu.addSeparator()
        self._show_grid_xy_action = QtGui.QAction("Show Grid (XY)", self)
        self._show_grid_xy_action.setCheckable(True)
        self._show_grid_xy_action.toggled.connect(self._on_overlay_action)
        view_menu.addAction(self._show_grid_xy_action)

        self._show_grid_xz_action = QtGui.QAction("Show Grid (XZ)", self)
        self._show_grid_xz_action.setCheckable(True)
        self._show_grid_xz_action.toggled.connect(self._on_overlay_action)
        view_menu.addAction(self._show_grid_xz_action)

        self._show_grid_yz_action = QtGui.QAction("Show Grid (YZ)", self)
        self._show_grid_yz_action.setCheckable(True)
        self._show_grid_yz_action.toggled.connect(self._on_overlay_action)
        view_menu.addAction(self._show_grid_yz_action)

        view_menu.addSeparator()
        orientation_menu = view_menu.addMenu("Orientation")
        self._orientation_actions: dict[str, QtGui.QAction] = {}
        for label, key in (
            ("-X", "-x"),
            ("X", "x"),
            ("-Y", "-y"),
            ("Y", "y"),
            ("-Z", "-z"),
            ("Z", "z"),
        ):
            action = QtGui.QAction(label, self)
            action.triggered.connect(lambda _checked=False, k=key: self._set_orientation(k))
            orientation_menu.addAction(action)
            self._orientation_actions[key] = action

        view_menu.addSeparator()
        self._spacecraft_editor_window_action = QtGui.QAction("Spacecraft Editor", self)
        self._spacecraft_editor_window_action.setCheckable(True)
        self._spacecraft_editor_window_action.setChecked(True)
        self._spacecraft_editor_window_action.toggled.connect(
            lambda checked: self._toggle_dock(self._builder_dock, checked)
        )
        self._builder_dock.visibilityChanged.connect(self._spacecraft_editor_window_action.setChecked)
        view_menu.addAction(self._spacecraft_editor_window_action)

        self._simulation_window_action = QtGui.QAction("Simulation", self)
        self._simulation_window_action.setCheckable(True)
        self._simulation_window_action.setChecked(True)
        self._simulation_window_action.toggled.connect(
            lambda checked: self._toggle_dock(self._simulation_dock, checked)
        )
        self._simulation_dock.visibilityChanged.connect(self._simulation_window_action.setChecked)
        view_menu.addAction(self._simulation_window_action)

        self._inspector_window_action = QtGui.QAction("Inspector", self)
        self._inspector_window_action.setCheckable(True)
        self._inspector_window_action.setChecked(False)
        self._inspector_window_action.toggled.connect(
            lambda checked: self._toggle_dock(self._inspector_dock, checked)
        )
        self._inspector_dock.visibilityChanged.connect(self._inspector_window_action.setChecked)
        view_menu.addAction(self._inspector_window_action)

        self._invariants_window_action = QtGui.QAction("Invariants / Truth Meter", self)
        self._invariants_window_action.setCheckable(True)
        self._invariants_window_action.setChecked(False)
        self._invariants_window_action.toggled.connect(
            lambda checked: self._toggle_dock(self._invariants_dock, checked)
        )
        self._invariants_dock.visibilityChanged.connect(self._invariants_window_action.setChecked)
        view_menu.addAction(self._invariants_window_action)
        self._inspector_dock.hide()
        self._invariants_dock.hide()

        self._interaction_mode = "select"
        self._interaction_toolbar = QtWidgets.QToolBar("Transform", self)
        self._interaction_toolbar.setMovable(False)
        self._interaction_toolbar.setOrientation(QtCore.Qt.Vertical)
        self._interaction_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._interaction_toolbar.setIconSize(QtCore.QSize(22, 22))
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self._interaction_toolbar)
        self._interaction_group = QtGui.QActionGroup(self)
        self._interaction_group.setExclusive(True)

        icon_root = Path("assets") / "icons"
        select_icon = QtGui.QIcon(str(icon_root / "select.svg"))
        move_icon = QtGui.QIcon(str(icon_root / "move.svg"))
        rotate_icon = QtGui.QIcon(str(icon_root / "rotate.svg"))

        self._select_tool_action = QtGui.QAction("Select", self)
        self._select_tool_action.setCheckable(True)
        self._select_tool_action.setChecked(True)
        self._select_tool_action.setIcon(select_icon)
        self._select_tool_action.triggered.connect(lambda: self._set_interaction_mode("select"))
        self._interaction_group.addAction(self._select_tool_action)
        self._interaction_toolbar.addAction(self._select_tool_action)

        self._move_tool_action = QtGui.QAction("Move", self)
        self._move_tool_action.setCheckable(True)
        self._move_tool_action.setIcon(move_icon)
        self._move_tool_action.triggered.connect(lambda: self._set_interaction_mode("move"))
        self._interaction_group.addAction(self._move_tool_action)
        self._interaction_toolbar.addAction(self._move_tool_action)

        self._rotate_tool_action = QtGui.QAction("Rotate", self)
        self._rotate_tool_action.setCheckable(True)
        self._rotate_tool_action.setIcon(rotate_icon)
        self._rotate_tool_action.triggered.connect(lambda: self._set_interaction_mode("rotate"))
        self._interaction_group.addAction(self._rotate_tool_action)
        self._interaction_toolbar.addAction(self._rotate_tool_action)

        self._connect_renderer(self._renderer)
        self._inspector.point_mass_updated.connect(self._on_point_mass_updated)
        self._inspector.rigid_body_updated.connect(self._on_rigid_body_updated)
        self._inspector.rigid_component_selected.connect(self._on_rigid_component_selected)
        self._builder_panel.mesh_load_requested.connect(self._on_mesh_load_requested)
        self._builder_panel.display_options_changed.connect(self._on_display_options_changed)
        self._builder_panel.components_changed.connect(self._on_components_updated)
        self._builder_panel.component_selected.connect(self._on_builder_component_selected)
        self._builder_panel.recenter_requested.connect(self._on_recenter_requested)
        self._builder_panel.initial_state_changed.connect(self._on_state_updated)
        self._builder_panel.reset_spacecraft_requested.connect(self._on_reset_spacecraft_requested)
        self._builder_panel.auto_generate_requested.connect(self._on_auto_generate_requested)
        self._simulation_panel.settings_changed.connect(self._on_simulation_settings_changed)

        self._simulation: Simulation | None = None
        self._selected_entity_id: Optional[str] = None
        self._selected_component_id: Optional[str] = None
        self._frame_choice: FrameChoice = FrameChoice.WORLD
        self._overlays = OverlayOptions()
        self._display_options = DisplayOptions()
        self._frame_vectors = None

        self._builder_panel.set_mesh_loading_available(
            mesh_loading_available(),
            mesh_loading_error() or "Install mesh extras to enable model loading.",
        )

        self._sync_view_menu()
        self._show_grid_xy_action.setChecked(True)
        self._on_overlay_action()
        if renderer3d_available():
            self._renderer_3d_action.setEnabled(True)
            self._renderer_3d_action.setToolTip("")
        else:
            self._renderer_3d_action.setEnabled(False)
            self._renderer_3d_action.setToolTip(
                renderer3d_error() or "Install 3D extras (PyOpenGL) to enable 3D mode."
            )
        default_renderer = "3d" if renderer3d_available() else "2d"
        self._on_renderer_changed(default_renderer, warn=False)
        self._interaction_toolbar.setVisible(Renderer3D is not None and isinstance(self._renderer, Renderer3D))

        self._new_scenario_from_defaults()

    def _on_scenario_action(self) -> None:
        action = self.sender()
        if not isinstance(action, QtGui.QAction):
            return
        scenario_id = action.data()
        if scenario_id is None:
            return
        scenario = scenario_registry.get(str(scenario_id))
        sim = scenario.create_simulation()
        self._scenario_controller.load_from_simulation(scenario.name, sim)
        self._set_simulation(self._scenario_controller.simulation)
        self._sync_scenario_ui()
        defaults = scenario.ui_defaults()
        if defaults and defaults.view_range:
            self._renderer.set_view_range(defaults.view_range)
        self._frame_scene_if_needed()

    def _set_simulation(self, sim: Simulation | None) -> None:
        if sim is None:
            return
        if self._timer.isActive():
            self._timer.stop()
            self._play_action.setChecked(False)
            self._play_action.setText("Play")
        self._simulation = sim
        self._selected_entity_id = None
        self._selected_component_id = None
        self._update_ui()

    def _new_scenario_from_defaults(self) -> None:
        self._scenario_controller.new_scenario("Untitled Scenario")
        self._set_simulation(self._scenario_controller.simulation)
        self._sync_scenario_ui()
        self._select_scenario_in_menu(None)

    def _new_scenario(self) -> None:
        dialog = NewScenarioDialog(self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        name, dt, integrator = dialog.values()
        self._scenario_controller.new_scenario(name, dt, integrator)
        self._set_simulation(self._scenario_controller.simulation)
        self._sync_scenario_ui()
        self._select_scenario_in_menu(None)

    def _open_scenario(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Scenario",
            "",
            "Scenario Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        self._scenario_controller.load_scenario(Path(file_path))
        self._set_simulation(self._scenario_controller.simulation)
        self._sync_scenario_ui()
        self._select_scenario_in_menu(None)

    def _save_scenario(self) -> None:
        try:
            self._scenario_controller.save_scenario()
        except RuntimeError:
            self._save_scenario_as()
        self._sync_scenario_ui()

    def _save_scenario_as(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Scenario As",
            "scenario.json",
            "Scenario Files (*.json);;All Files (*)",
        )
        if not file_path:
            return
        self._scenario_controller.save_scenario(Path(file_path))
        self._sync_scenario_ui()

    def _on_add_spacecraft_requested(self) -> None:
        body_def = self._scenario_controller.add_spacecraft()
        self._selected_entity_id = body_def.entity_id
        self._selected_component_id = None
        self._sync_scenario_ui()
        self._update_ui()

    def _on_simulation_settings_changed(self, dt: float, integrator: str) -> None:
        self._scenario_controller.update_simulation_settings(dt, integrator)
        self._sync_scenario_ui()
        if self._timer.isActive() and self._simulation is not None:
            interval_ms = int(self._simulation.dt * 1000)
            self._timer.start(max(interval_ms, 1))

    def _open_scenario_settings(self) -> None:
        scenario = self._scenario_controller.scenario
        if scenario is None:
            return
        dialog = ScenarioSettingsDialog(
            scenario.name,
            scenario.simulation.dt,
            scenario.simulation.integrator,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        name, dt, integrator = dialog.values()
        self._scenario_controller.update_scenario_name(name)
        self._scenario_controller.update_simulation_settings(dt, integrator)
        self._sync_scenario_ui()

    def _sync_scenario_ui(self) -> None:
        scenario = self._scenario_controller.scenario
        if scenario is None:
            return
        self._simulation_panel.set_settings(scenario.simulation.dt, scenario.simulation.integrator)

    def _update_ui(self) -> None:
        if self._simulation is None:
            return
        self._frame_vectors = compute_frame_vectors(self._simulation.entities, self._frame_choice)
        self._renderer.set_scene(
            self._frame_vectors,
            self._overlays,
            self._display_options,
            self._selected_entity_id,
            self._selected_component_id,
            entities=self._simulation.entities,
        )
        self._invariants.update_values(self._simulation.entities)
        entity, _ = self._resolve_entity(self._selected_entity_id)
        self._inspector.set_entity(entity)
        if isinstance(entity, RigidBody):
            self._builder_panel.set_entity(entity)
            component_id = self._selected_component_id_for_entity(entity)
            self._builder_panel.set_selected_component(component_id)
        else:
            self._builder_panel.set_entity(None)

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
        entity, component = self._resolve_entity(entity_id)
        if component is not None and isinstance(entity, RigidBody):
            self._selected_entity_id = entity.entity_id
            self._selected_component_id = entity_id
        else:
            self._selected_entity_id = entity_id
            self._selected_component_id = None
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
        self._scenario_controller.update_point_mass(
            entity.entity_id,
            entity.mass,
            entity.position,
            entity.velocity,
        )
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
        position_frame = np.array(position, dtype=float)
        if self._frame_vectors is not None:
            position_vec = np.zeros(self._frame_vectors.dimension, dtype=float)
            position_vec[:2] = position_frame[:2]
            position_world = from_frame(position_vec, self._frame_choice, self._frame_vectors.r_oc_world)
        else:
            position_world = position_frame
        self._scenario_controller.update_point_mass(
            entity_id,
            mass,
            position_world,
            np.array(velocity, dtype=float),
        )
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
        orientation = tuple(float(value) for value in entity.orientation)
        self._scenario_controller.update_state(
            entity_id,
            com_position,
            com_velocity,
            orientation,
            omega_world,
        )
        self._update_ui()

    def _on_rigid_component_selected(self, entity_id: str, component_id: object) -> None:
        if component_id is None:
            self._selected_component_id = None
            self._update_ui()
            return
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            return
        self._selected_entity_id = entity_id
        self._selected_component_id = entity.component_entity_id(str(component_id))
        self._update_ui()

    def _on_builder_component_selected(self, entity_id: str, component_id: object) -> None:
        if component_id is None:
            self._selected_component_id = None
            self._update_ui()
            return
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            return
        self._selected_entity_id = entity_id
        self._selected_component_id = entity.component_entity_id(str(component_id))
        self._update_ui()

    def _on_display_options_changed(self, options: DisplayOptions) -> None:
        self._display_options = options
        self._update_ui()

    def _on_components_updated(self, entity_id: str, components: list[RigidBodyComponent]) -> None:
        if not components:
            return
        self._scenario_controller.update_components(entity_id, components)
        entity, _ = self._resolve_entity(entity_id)
        if entity is not None and isinstance(entity, RigidBody) and self._selected_component_id is not None:
            valid_ids = {entity.component_entity_id(comp.component_id) for comp in entity.components}
            if self._selected_component_id not in valid_ids:
                self._selected_component_id = None
        self._update_ui()

    def _on_state_updated(
        self,
        entity_id: str,
        com_position: tuple[float, float, float],
        com_velocity: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
        omega_world: tuple[float, float, float],
    ) -> None:
        self._scenario_controller.update_state(
            entity_id,
            com_position,
            com_velocity,
            orientation,
            omega_world,
        )
        self._update_ui()

    def _on_reset_spacecraft_requested(self, entity_id: str) -> None:
        self._scenario_controller.reset_spacecraft(entity_id)
        self._selected_component_id = None
        self._update_ui()

    def _on_auto_generate_requested(self, entity_id: str, max_points: int) -> None:
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            return
        if entity.mesh is None or not entity.mesh.path:
            QtWidgets.QMessageBox.information(
                self,
                "Auto-generate Mass Points",
                "Load a mesh model first to auto-generate mass points.",
            )
            return
        if not mesh_loading_available():
            QtWidgets.QMessageBox.information(
                self,
                "Mesh Loading Unavailable",
                "Install mesh extras to enable auto generation (pip install -e .[mesh]).",
            )
            return
        components = self._generate_components_from_mesh(entity, max_points=max_points)
        if not components:
            QtWidgets.QMessageBox.information(
                self,
                "Auto-generate Mass Points",
                "Unable to auto-generate mass points from the mesh.",
            )
            return
        self._on_components_updated(entity_id, components)

    def _on_recenter_requested(self, entity_id: str) -> None:
        self._scenario_controller.recenter_components(entity_id)
        self._update_ui()

    def _on_mesh_load_requested(self) -> None:
        entity, _ = self._resolve_entity(self._selected_entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            QtWidgets.QMessageBox.information(
                self, "Spacecraft Editor", "Select a RigidBody to attach a model."
            )
            return
        if not mesh_loading_available():
            QtWidgets.QMessageBox.information(
                self,
                "Mesh Loading Unavailable",
                "Install mesh extras to enable model loading (pip install -e .[mesh]).",
            )
            return
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Model",
            "",
            "Mesh Files (*.glb *.gltf *.obj *.stl);;All Files (*)",
        )
        if not file_path:
            return
        mesh_meta, info = self._mesh_metadata_from_path(file_path)
        self._scenario_controller.set_mesh_metadata(entity.entity_id, mesh_meta)
        self._builder_panel.set_mesh_info(info)
        self._update_ui()

    def _reset_scene(self) -> None:
        self._scenario_controller.reset_simulation()
        self._set_simulation(self._scenario_controller.simulation)
        self._frame_scene_if_needed()

    def _apply_scene(self, scene: SceneData) -> None:
        scene_copy = clone_scene(scene)
        scenario_id = scene_copy.scenario_id
        if scenario_id and scenario_id in {s.scenario_id for s in scenario_registry.all()}:
            scenario = scenario_registry.get(scenario_id)
            sim = scenario.create_simulation()
            sim.entities = scene_copy.entities
        else:
            sim = Simulation(entities=scene_copy.entities, dt=0.05, integrator=SymplecticEulerIntegrator())
        self._scenario_controller.load_from_simulation("Loaded Scene", sim)
        self._set_simulation(self._scenario_controller.simulation)
        self._sync_scenario_ui()
        self._select_scenario_in_menu(scenario_id if scenario_id in self._scenario_actions else None)
        self._frame_scene_if_needed()

    def _select_scenario_in_menu(self, scenario_id: str | None) -> None:
        if scenario_id is None or scenario_id not in self._scenario_actions:
            self._scenario_group.blockSignals(True)
            for action in self._scenario_group.actions():
                action.setChecked(False)
            self._scenario_group.blockSignals(False)
            return
        action = self._scenario_actions[scenario_id]
        self._scenario_group.blockSignals(True)
        action.setChecked(True)
        self._scenario_group.blockSignals(False)

    def _on_frame_action(self, frame: FrameChoice) -> None:
        self._frame_choice = frame
        self._set_frame_actions(frame)
        self._update_ui()

    def _on_overlay_action(self) -> None:
        self._overlays = OverlayOptions(
            show_r_op=self._show_r_op_action.isChecked(),
            show_r_oc=self._show_r_oc_action.isChecked(),
            show_r_cp=self._show_r_cp_action.isChecked(),
            show_axes=self._show_axes_action.isChecked(),
            show_axis_labels=self._show_axis_labels_action.isChecked(),
            show_grid_xy=self._show_grid_xy_action.isChecked(),
            show_grid_xz=self._show_grid_xz_action.isChecked(),
            show_grid_yz=self._show_grid_yz_action.isChecked(),
        )
        self._update_ui()

    def _on_renderer_changed(self, mode: str, warn: bool = True) -> None:
        if mode == "3d" and not renderer3d_available():
            if warn:
                QtWidgets.QMessageBox.information(
                    self,
                    "3D Mode Unavailable",
                    renderer3d_error() or "Install 3D extras (PyOpenGL) to enable 3D mode.",
                )
            self._set_renderer_actions("2d")
            return
        renderer = self._ensure_renderer3d() if mode == "3d" else self._renderer_2d
        self._swap_renderer(renderer)
        self._set_renderer_actions(mode)

    def _sync_view_menu(self) -> None:
        self._set_frame_actions(self._frame_choice)
        current_mode = "3d" if self._renderer is self._renderer_3d else "2d"
        self._set_renderer_actions(current_mode)
        self._show_axes_action.setChecked(self._overlays.show_axes)
        self._show_axis_labels_action.setChecked(self._overlays.show_axis_labels)
        self._show_r_op_action.setChecked(self._overlays.show_r_op)
        self._show_r_oc_action.setChecked(self._overlays.show_r_oc)
        self._show_r_cp_action.setChecked(self._overlays.show_r_cp)
        self._show_grid_xy_action.setChecked(self._overlays.show_grid_xy)
        self._show_grid_xz_action.setChecked(self._overlays.show_grid_xz)
        self._show_grid_yz_action.setChecked(self._overlays.show_grid_yz)

    def _set_frame_actions(self, frame: FrameChoice) -> None:
        self._frame_group.blockSignals(True)
        self._frame_world_action.setChecked(frame == FrameChoice.WORLD)
        self._frame_com_action.setChecked(frame == FrameChoice.COM)
        self._frame_group.blockSignals(False)

    def _set_renderer_actions(self, mode: str) -> None:
        self._renderer_group.blockSignals(True)
        self._renderer_2d_action.setChecked(mode == "2d")
        self._renderer_3d_action.setChecked(mode == "3d")
        self._renderer_group.blockSignals(False)

    def _set_interaction_mode(self, mode: str) -> None:
        self._interaction_mode = mode
        if Renderer3D is not None and isinstance(self._renderer, Renderer3D):
            self._renderer.set_interaction_mode(mode)

    def _on_transform_requested(self, entity_id: str, mode: str, payload: object) -> None:
        if self._simulation is None:
            return
        entity, component = resolve_entity_by_id(self._simulation.entities, entity_id)
        if entity is None:
            return
        if mode == "move":
            delta = np.asarray(payload, dtype=float).reshape(-1)
            if isinstance(entity, PointMass):
                if entity.position.size == 2:
                    entity.position = entity.position + delta[:2]
                else:
                    entity.position = entity.position + delta[: entity.position.size]
                self._update_ui()
                return
            if not isinstance(entity, RigidBody):
                return
            if component is None:
                entity.com_position = entity.com_position + delta[:3]
                self._update_ui()
                return
            rotation = entity.rotation_matrix()
            delta_body = rotation.T @ delta[:3]
            new_components = []
            for comp in entity.components:
                if comp.component_id == component.component_id:
                    new_components.append(
                        RigidBodyComponent(
                            component_id=comp.component_id,
                            mass=comp.mass,
                            position_body=comp.position_body + delta_body,
                        )
                    )
                else:
                    new_components.append(comp)
            entity.components = new_components
            self._update_ui()
            return
        if mode == "rotate":
            if not isinstance(entity, RigidBody):
                return
            if not isinstance(payload, dict):
                return
            axis = np.asarray(payload.get("axis", [0.0, 0.0, 1.0]), dtype=float).reshape(3)
            angle_deg = float(payload.get("angle_deg", 0.0))
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1e-9 or abs(angle_deg) <= 1e-9:
                return
            axis = axis / axis_norm
            q_delta = self._quat_from_axis_angle(axis, np.deg2rad(angle_deg))
            entity.orientation = self._normalize_quaternion(self._quat_multiply(q_delta, entity.orientation))
            self._update_ui()
            return

    def _set_orientation(self, key: str) -> None:
        if Renderer3D is not None and isinstance(self._renderer, Renderer3D):
            self._renderer.set_orientation(key)
            return
        QtWidgets.QMessageBox.information(
            self,
            "Orientation Unavailable",
            "Orientation controls are available in 3D mode only.",
        )

    @staticmethod
    def _toggle_dock(dock: QtWidgets.QDockWidget, visible: bool) -> None:
        if visible:
            dock.show()
            dock.raise_()
        else:
            dock.hide()

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
        self._frame_scene_if_needed()
        show_tools = Renderer3D is not None and isinstance(renderer, Renderer3D)
        self._interaction_toolbar.setVisible(show_tools)
        if show_tools:
            renderer.set_interaction_mode(self._interaction_mode)

    def _connect_renderer(self, renderer) -> None:
        renderer.entity_selected.connect(self._on_entity_selected)
        renderer.entity_dragged.connect(self._on_entity_dragged)
        if hasattr(renderer, "transform_requested"):
            renderer.transform_requested.connect(self._on_transform_requested)

    def _disconnect_renderer(self, renderer) -> None:
        try:
            renderer.entity_selected.disconnect(self._on_entity_selected)
        except (TypeError, RuntimeError):
            pass
        try:
            renderer.entity_dragged.disconnect(self._on_entity_dragged)
        except (TypeError, RuntimeError):
            pass
        if hasattr(renderer, "transform_requested"):
            try:
                renderer.transform_requested.disconnect(self._on_transform_requested)
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
        scene = capture_scene(self._simulation.entities, scenario_id=None)
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

    def _resolve_entity(
        self, entity_id: str | None
    ) -> tuple[PointMass | RigidBody | None, object | None]:
        if self._simulation is None or entity_id is None:
            return None, None
        return resolve_entity_by_id(self._simulation.entities, entity_id)

    def _frame_scene(self) -> None:
        self._renderer.frame_scene()

    def _frame_scene_if_needed(self) -> None:
        if self._renderer is self._renderer_2d:
            return
        self._renderer.frame_scene()

    def _selected_component_id_for_entity(self, entity: RigidBody) -> str | None:
        if self._selected_component_id is None:
            return None
        for component in entity.components:
            if entity.component_entity_id(component.component_id) == self._selected_component_id:
                return component.component_id
        return None

    def _mesh_metadata_from_path(self, file_path: str) -> tuple[MeshMetadata, MeshInfo]:
        path = Path(file_path).resolve()
        project_root = Path.cwd().resolve()
        assets_root = project_root / "assets"
        mesh_info = "Mesh info unavailable"
        warning = ""
        try:
            mesh_data = load_mesh_data(path)
            mesh_info = f"{mesh_data.vertex_count} vertices, {mesh_data.face_count} faces"
        except Exception as exc:
            mesh_info = f"Mesh loaded with warnings: {exc}"
        if self._is_under_root(path, assets_root):
            rel_path = path.relative_to(assets_root)
            mesh_path = Path("assets") / rel_path
            metadata = MeshMetadata(path=str(mesh_path).replace("\\", "/"), path_is_absolute=False)
        else:
            metadata = MeshMetadata(path=str(path), path_is_absolute=True)
            warning = "External path (absolute) reduces portability."
        return metadata, self._mesh_info_from_metadata(metadata, mesh_info, warning)

    @staticmethod
    def _is_under_root(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _mesh_info_from_metadata(metadata: MeshMetadata, info: str, warning: str) -> MeshInfo:
        return MeshInfo(path=metadata.path, info=info, warning=warning)


    @staticmethod
    def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=float).reshape(4)
        norm = float(np.linalg.norm(q_arr))
        if norm <= 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q_arr / norm

    @staticmethod
    def _quat_multiply(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = np.asarray(q_left, dtype=float).reshape(4)
        w2, x2, y2, z2 = np.asarray(q_right, dtype=float).reshape(4)
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    @staticmethod
    def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=float).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        axis = axis / norm
        half = 0.5 * angle_rad
        return np.array(
            [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
            dtype=float,
        )

    def _generate_components_from_mesh(self, entity: RigidBody, max_points: int = 5) -> list[RigidBodyComponent]:
        if entity.mesh is None or not entity.mesh.path:
            return []
        mesh_path = Path(entity.mesh.path)
        if not mesh_path.is_absolute():
            mesh_path = (Path.cwd() / mesh_path).resolve()
        try:
            mesh_data = load_mesh_data(mesh_path)
        except Exception:
            return []
        vertices = self._apply_mesh_transform(mesh_data.vertices, entity.mesh)
        faces = mesh_data.faces if hasattr(mesh_data, "faces") else None
        points = generate_mass_points_from_mesh(vertices, faces, max_points=max(1, int(max_points)))
        components = [
            RigidBodyComponent(
                component_id=f"P{i + 1}",
                mass=1.0,
                position_body=np.array(point, dtype=float),
            )
            for i, point in enumerate(points)
        ]
        return components

    @staticmethod
    def _apply_mesh_transform(vertices: np.ndarray, mesh: MeshMetadata) -> np.ndarray:
        scaled = vertices * mesh.scale.reshape(1, 3)
        rotation = MainWindow._quat_to_matrix(mesh.rotation_body)
        rotated = (rotation @ scaled.T).T
        return rotated + mesh.offset_body.reshape(1, 3)

    @staticmethod
    def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = np.asarray(q, dtype=float).reshape(4)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )
