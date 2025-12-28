from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.model import MeshMetadata, PointMass, RigidBody, RigidBodyComponent, resolve_entity_by_id
from ..core.scenario_definition import (
    EntityDefinition,
    PointMassDefinition,
    RigidBodyComponentDefinition,
    RigidBodyDefinition,
    RigidBodyStateDefinition,
    ScenarioDefinition,
    SimulationDefinition,
    definition_from_simulation,
    integrator_from_id,
    simulation_from_definition,
)
from ..core.io.scenario_io import deserialize_scenario_definition, serialize_scenario_definition
from ..core.sim import Simulation


@dataclass
class ScenarioContext:
    definition: ScenarioDefinition
    simulation: Simulation
    path: Path | None = None


class ScenarioController:
    def __init__(self) -> None:
        self._context: ScenarioContext | None = None

    @property
    def scenario(self) -> ScenarioDefinition | None:
        return None if self._context is None else self._context.definition

    @property
    def simulation(self) -> Simulation | None:
        return None if self._context is None else self._context.simulation

    @property
    def scenario_path(self) -> Path | None:
        return None if self._context is None else self._context.path

    def new_scenario(self, name: str, dt: float = 0.05, integrator: str = "symplectic_euler") -> None:
        definition = ScenarioDefinition(
            name=name.strip() or "Untitled Scenario",
            simulation=SimulationDefinition(dt=float(dt), integrator=integrator),
            created_at=self._timestamp(),
            modified_at=self._timestamp(),
        )
        self._context = ScenarioContext(definition=definition, simulation=simulation_from_definition(definition))

    def load_scenario(self, path: Path) -> None:
        payload = path.read_text(encoding="utf-8")
        definition = deserialize_scenario_definition(payload)
        self._context = ScenarioContext(
            definition=definition,
            simulation=simulation_from_definition(definition),
            path=path,
        )

    def save_scenario(self, path: Path | None = None) -> Path:
        if self._context is None:
            raise RuntimeError("No scenario loaded")
        if path is not None:
            self._context.path = path
        if self._context.path is None:
            raise RuntimeError("Scenario path is not set")
        self._context.definition.modified_at = self._timestamp()
        payload = serialize_scenario_definition(self._context.definition)
        self._context.path.write_text(payload, encoding="utf-8")
        return self._context.path

    def load_from_simulation(self, name: str, sim: Simulation) -> None:
        definition = definition_from_simulation(name, sim)
        self._context = ScenarioContext(definition=definition, simulation=simulation_from_definition(definition))

    def update_scenario_name(self, name: str) -> None:
        if self._context is None:
            return
        self._context.definition.name = name.strip() or "Untitled Scenario"
        self._context.definition.modified_at = self._timestamp()

    def update_simulation_settings(self, dt: float, integrator: str) -> None:
        if self._context is None:
            return
        definition = self._context.definition
        definition.simulation.dt = float(dt)
        definition.simulation.integrator = integrator
        definition.modified_at = self._timestamp()
        sim = self._context.simulation
        sim.dt = float(dt)
        sim.integrator = integrator_from_id(integrator)

    def reset_simulation(self) -> None:
        if self._context is None:
            return
        self._context.simulation = simulation_from_definition(self._context.definition)

    def add_spacecraft(self) -> RigidBodyDefinition:
        if self._context is None:
            raise RuntimeError("No scenario loaded")
        definition = self._context.definition
        entity_id = self._next_spacecraft_id(definition.entities)
        components = [
            RigidBodyComponentDefinition(
                component_id="C1",
                mass=1.0,
                position_body=np.zeros(3, dtype=float),
            )
        ]
        state = RigidBodyStateDefinition()
        body_def = RigidBodyDefinition(entity_id=entity_id, components=components, state=state, mesh=None)
        definition.entities.append(body_def)
        definition.modified_at = self._timestamp()
        self._context.simulation.entities.append(self._instantiate_rigid_body(body_def))
        return body_def

    def update_components(self, entity_id: str, components: Iterable[RigidBodyComponent]) -> None:
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            return
        components_list = list(components)
        if not components_list:
            return
        new_com_body = self._compute_com_body(components_list)
        rotation = entity.rotation_matrix()
        if np.linalg.norm(new_com_body) > 1e-12:
            components_list = [
                RigidBodyComponent(
                    component_id=component.component_id,
                    mass=component.mass,
                    position_body=component.position_body - new_com_body,
                )
                for component in components_list
            ]
            entity.com_position = entity.com_position + rotation @ new_com_body
            if entity.mesh is not None:
                entity.mesh.offset_body = entity.mesh.offset_body - new_com_body
        entity.components = components_list
        self._sync_definition_components(entity, components_list)

    def update_state(
        self,
        entity_id: str,
        com_position: tuple[float, float, float],
        com_velocity: tuple[float, float, float],
        orientation: tuple[float, float, float, float],
        omega_world: tuple[float, float, float],
    ) -> None:
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            return
        entity.com_position = np.array(com_position, dtype=float)
        entity.com_velocity = np.array(com_velocity, dtype=float)
        entity.orientation = self._normalize_quaternion(np.array(orientation, dtype=float))
        entity.omega_world = np.array(omega_world, dtype=float)
        self._sync_definition_state(entity)

    def recenter_components(self, entity_id: str) -> None:
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            return
        masses = np.array([component.mass for component in entity.components], dtype=float)
        positions = np.stack([component.position_body for component in entity.components])
        com_body = np.sum(positions * masses[:, None], axis=0) / np.sum(masses)
        if np.allclose(com_body, np.zeros(3), atol=1e-12):
            return
        entity.components = [
            RigidBodyComponent(
                component_id=component.component_id,
                mass=component.mass,
                position_body=component.position_body - com_body,
            )
            for component in entity.components
        ]
        if entity.mesh is not None:
            entity.mesh.offset_body = entity.mesh.offset_body - com_body
        self._sync_definition_components(entity, entity.components)

    def set_mesh_metadata(self, entity_id: str, mesh: MeshMetadata | None) -> None:
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            return
        entity.mesh = mesh
        self._sync_definition_mesh(entity)

    def reset_spacecraft(self, entity_id: str) -> None:
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, RigidBody):
            return
        entity.components = [
            RigidBodyComponent(component_id="C1", mass=1.0, position_body=np.zeros(3, dtype=float))
        ]
        entity.com_position = np.zeros(3, dtype=float)
        entity.com_velocity = np.zeros(3, dtype=float)
        entity.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        entity.omega_world = np.zeros(3, dtype=float)
        entity.mesh = None
        self._sync_definition_components(entity, entity.components)
        self._sync_definition_state(entity)
        self._sync_definition_mesh(entity)

    def update_point_mass(
        self,
        entity_id: str,
        mass: float,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        entity, _ = self._resolve_entity(entity_id)
        if entity is None or not isinstance(entity, PointMass):
            return
        entity.mass = float(mass)
        entity.position = np.array(position, dtype=float)
        entity.velocity = np.array(velocity, dtype=float)
        point_def = self._find_point_mass_definition(entity_id)
        if point_def is None:
            return
        point_def.mass = float(mass)
        point_def.position = np.array(position, dtype=float)
        point_def.velocity = np.array(velocity, dtype=float)
        if self._context is not None:
            self._context.definition.modified_at = self._timestamp()

    def _resolve_entity(self, entity_id: str) -> tuple[RigidBody | None, object | None]:
        if self._context is None:
            return None, None
        return resolve_entity_by_id(self._context.simulation.entities, entity_id)

    def _sync_definition_components(self, entity: RigidBody, components: Iterable[RigidBodyComponent]) -> None:
        if self._context is None:
            return
        body_def = self._find_definition(entity.entity_id)
        if body_def is None:
            return
        body_def.components = [
            RigidBodyComponentDefinition(
                component_id=component.component_id,
                mass=float(component.mass),
                position_body=np.array(component.position_body, dtype=float),
            )
            for component in components
        ]
        body_def.state.com_position = np.array(entity.com_position, dtype=float)
        if entity.mesh is not None:
            body_def.mesh = MeshMetadata(
                path=str(entity.mesh.path),
                path_is_absolute=bool(entity.mesh.path_is_absolute),
                scale=np.array(entity.mesh.scale, dtype=float),
                offset_body=np.array(entity.mesh.offset_body, dtype=float),
                rotation_body=np.array(entity.mesh.rotation_body, dtype=float),
            )
        self._context.definition.modified_at = self._timestamp()

    def _sync_definition_state(self, entity: RigidBody) -> None:
        if self._context is None:
            return
        body_def = self._find_definition(entity.entity_id)
        if body_def is None:
            return
        body_def.state = RigidBodyStateDefinition(
            com_position=np.array(entity.com_position, dtype=float),
            com_velocity=np.array(entity.com_velocity, dtype=float),
            orientation=np.array(entity.orientation, dtype=float),
            omega_world=np.array(entity.omega_world, dtype=float),
        )
        self._context.definition.modified_at = self._timestamp()

    def _sync_definition_mesh(self, entity: RigidBody) -> None:
        if self._context is None:
            return
        body_def = self._find_definition(entity.entity_id)
        if body_def is None:
            return
        if entity.mesh is None:
            body_def.mesh = None
        else:
            body_def.mesh = MeshMetadata(
                path=str(entity.mesh.path),
                path_is_absolute=bool(entity.mesh.path_is_absolute),
                scale=np.array(entity.mesh.scale, dtype=float),
                offset_body=np.array(entity.mesh.offset_body, dtype=float),
                rotation_body=np.array(entity.mesh.rotation_body, dtype=float),
            )
        self._context.definition.modified_at = self._timestamp()

    def _find_definition(self, entity_id: str) -> RigidBodyDefinition | None:
        if self._context is None:
            return None
        for entity in self._context.definition.entities:
            if entity.entity_id == entity_id:
                if isinstance(entity, RigidBodyDefinition):
                    return entity
        return None

    def _find_point_mass_definition(self, entity_id: str) -> PointMassDefinition | None:
        if self._context is None:
            return None
        for entity in self._context.definition.entities:
            if entity.entity_id == entity_id and isinstance(entity, PointMassDefinition):
                return entity
        return None

    @staticmethod
    def _compute_com_body(components: Iterable[RigidBodyComponent]) -> np.ndarray:
        components_list = list(components)
        masses = np.array([component.mass for component in components_list], dtype=float)
        positions = np.stack([component.position_body for component in components_list])
        return np.sum(positions * masses[:, None], axis=0) / np.sum(masses)

    @staticmethod
    def _next_spacecraft_id(entities: Iterable[EntityDefinition]) -> str:
        used = {entity.entity_id for entity in entities}
        index = 1
        while f"SC-{index}" in used:
            index += 1
        return f"SC-{index}"

    @staticmethod
    def _instantiate_rigid_body(defn: RigidBodyDefinition) -> RigidBody:
        return RigidBody(
            entity_id=defn.entity_id,
            components=[
                RigidBodyComponent(
                    component_id=component.component_id,
                    mass=float(component.mass),
                    position_body=np.array(component.position_body, dtype=float),
                )
                for component in defn.components
            ],
            com_position=np.array(defn.state.com_position, dtype=float),
            com_velocity=np.array(defn.state.com_velocity, dtype=float),
            orientation=np.array(defn.state.orientation, dtype=float),
            omega_world=np.array(defn.state.omega_world, dtype=float),
            mesh=defn.mesh,
        )

    @staticmethod
    def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q, dtype=float).reshape(4)
        norm = float(np.linalg.norm(q_arr))
        if norm <= 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q_arr / norm

    @staticmethod
    def _timestamp() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
